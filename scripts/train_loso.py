import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import yaml
import json
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from src.data.preprocessing import IMUPreprocessor
from src.data.windowing import WindowGenerator
from src.models.lstm_classifier import LSTMClassifier


class LOSOTrainer:
    """Pipeline LOSO pour classification frame-wise"""
    
    def __init__(self, config_path='config/config.yaml'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Paths
        self.data_dir = Path(self.config['paths']['processed_data'])
        self.models_dir = Path(self.config['paths']['models_dir'])
        self.models_dir.mkdir(exist_ok=True)
        
        # Parameters
        self.window_size = self.config['data']['window_size']
        self.overlap = self.config['data']['overlap']
        self.num_classes = len(self.config['data']['label_mapping'])
        
        # Model hyperparameters
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.2
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 50
        self.patience = 10
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
    
    def load_subject_data(self, subject_id):
        """Charge les données d'un sujet"""
        file_path = self.data_dir / f"{subject_id}.csv"
        df = pd.read_csv(file_path)
        
        # Feature columns basées sur votre CSV
        # Exclure: time, label, label_idx
        exclude_cols = ['time', 'label', 'label_idx']
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['label_idx'].values  # Utiliser label_idx au lieu de label
        
        return X, y, feature_cols
    
    def compute_class_weights(self, y_train_list):
        """Calcule les poids de classes pour l'équilibrage"""
        # Concaténer toutes les étiquettes d'entraînement
        y_flat = np.concatenate([y.flatten() for y in y_train_list])
        
        classes = np.unique(y_flat)
        weights = compute_class_weight('balanced', classes=classes, y=y_flat)
        
        return torch.FloatTensor(weights)
    
    def train_epoch(self, model, dataloader, criterion, optimizer):
        """Entraînement d'une epoch"""
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)  # (batch, seq_len, features)
            y_batch = y_batch.to(self.device)  # (batch, seq_len)
            
            optimizer.zero_grad()
            
            # Forward: frame-wise classification
            logits = model(X_batch)  # (batch, seq_len, num_classes)
            
            # Loss frame-wise
            loss = criterion(
                logits.reshape(-1, self.num_classes),
                y_batch.reshape(-1)
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, model, dataloader):
        """Évaluation du modèle"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits = model(X_batch)  # (batch, seq_len, num_classes)
                
                # Prédiction: argmax sur dim des classes
                preds = torch.argmax(logits, dim=-1)  # (batch, seq_len)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())
        
        # Flatten toutes les prédictions et labels
        all_preds = np.concatenate([p.flatten() for p in all_preds])
        all_labels = np.concatenate([l.flatten() for l in all_labels])
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        
        return accuracy, f1_macro, all_preds, all_labels
    
    def train_fold(self, test_subject, all_subjects):
        """Entraîne un fold LOSO"""
        print(f"\n{'='*60}")
        print(f"FOLD: Test Subject = {test_subject}")
        print(f"{'='*60}")
        
        # 1. Définir train subjects
        train_subjects = [s for s in all_subjects if s != test_subject]
        print(f"Train subjects: {train_subjects}")
        
        # 2. Charger les données
        train_data = {}
        for subj in train_subjects:
            X, y, feature_cols = self.load_subject_data(subj)
            train_data[subj] = {'X': X, 'y': y}
        
        X_test, y_test, _ = self.load_subject_data(test_subject)
        
        # 3. Fit preprocessor sur TRAIN uniquement
        preprocessor = IMUPreprocessor(
            method="zscore",
            use_lowpass=True,      # recommended
            cutoff_hz=25.0,
            sampling_rate=100
        )
        for subj in train_subjects:
            preprocessor.fit(train_data[subj]['X'], subject_id=subj)
        
        # 4. Transform train et test
        windower = WindowGenerator(
            window_size=self.window_size,
            overlap=self.overlap
        )
        
        X_train_list = []
        y_train_list = []
        
        for subj in train_subjects:
            X_norm = preprocessor.transform(train_data[subj]['X'], subject_id=subj)
            
            # Windowing avec sequence labeling
            X_win, y_win = windower.create_windows_sequence_labeling(
                X_norm, train_data[subj]['y']
            )
            
            X_train_list.append(X_win)
            y_train_list.append(y_win)
        
        # Concaténer tous les sujets d'entraînement
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        
        # Transform test
        X_test_norm = preprocessor.transform(X_test, subject_id=test_subject)
        X_test, y_test = windower.create_windows_sequence_labeling(X_test_norm, y_test)
        
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # 5. Calculer class weights
        class_weights = self.compute_class_weights(y_train_list)
        print(f"Class weights: {class_weights}")
        
        # 6. Créer DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 7. Initialiser le modèle
        input_size = X_train.shape[2]  # Nombre de features
        model = LSTMClassifier(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss avec class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # 8. Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            val_acc, val_f1, _, _ = self.evaluate(model, test_loader)
            
            print(f"Epoch {epoch+1:3d}/{self.epochs}: "
                  f"Loss: {train_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}, "
                  f"Val F1: {val_f1:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'test_subject': test_subject
                }, self.models_dir / f'model_{test_subject}.pth')
                
                print(f"  ✓ Best model saved (acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            scheduler.step(train_loss)
        
        # 9. Évaluation finale
        checkpoint = torch.load(self.models_dir / f'model_{test_subject}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        final_acc, final_f1, preds, labels = self.evaluate(model, test_loader)
        
        print(f"\nFinal Results for {test_subject}:")
        print(f"  Accuracy: {final_acc:.4f}")
        print(f"  F1 Macro: {final_f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(labels, preds, 
                                   target_names=[f"Class_{i}" for i in range(self.num_classes)]))
        
        return {
            'test_subject': test_subject,
            'accuracy': float(final_acc),
            'f1_macro': float(final_f1),
            'num_train_windows': len(X_train),
            'num_test_windows': len(X_test)
        }
    
    def run_loso(self):
        """Exécute LOSO cross-validation sur tous les sujets"""
        # Récupérer tous les sujets depuis les fichiers CSV
        all_files = list(self.data_dir.glob('*.csv'))
        all_subjects = sorted([f.stem for f in all_files])

        
        print(f"Found {len(all_subjects)} subjects: {all_subjects}")
        print(f"Expected format: S01, S02, ..., S10")
        
        # Entraîner chaque fold
        all_results = []
        
        for test_subject in all_subjects:
            results = self.train_fold(test_subject, all_subjects)
            all_results.append(results)
        
        # Agréger les résultats
        accuracies = [r['accuracy'] for r in all_results]
        f1_scores = [r['f1_macro'] for r in all_results]
        
        print(f"\n{'='*60}")
        print("LOSO CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Mean F1 Macro: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"{'='*60}")
        
        # Sauvegarder les résultats
        final_results = {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'mean_f1_macro': float(np.mean(f1_scores)),
            'std_f1_macro': float(np.std(f1_scores)),
            'fold_results': all_results,
            'config': {
                'window_size': self.window_size,
                'overlap': self.overlap,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }
        
        with open(self.models_dir / 'loso_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\nResults saved to {self.models_dir / 'loso_results.json'}")


if __name__ == "__main__":
    trainer = LOSOTrainer()
    trainer.run_loso()