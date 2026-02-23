import yaml
import time
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from src.data.preprocessing import IMUPreprocessor
from src.data.windowing import WindowGenerator
from src.models.factory import create_cnn_lstm_model
from src.models.lstm_frame import LSTMClassifier
from src.models.cnn_lstm import CNNLSTMClassifier, CNNLSTMClassifierDeep


class AblationStudyTrainer:
    """
    LOSO Trainer with ablation study support.
    Tests different hyperparameter configurations systematically.
    """

    def __init__(self, config_path='/kaggle/input/datasets/manelouchar/config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = Path("/kaggle/input/datasets/manelouchar/processed")

        # Base output directory
        self.base_dir = Path('/kaggle/working/ablation_study')
        self.base_dir.mkdir(exist_ok=True)

        # Core hyperparameters (fixed unless ablated)
        self.num_classes = len(self.config['data']['classes'])
        self.hidden_size = 256
        self.num_layers = 2
        self.bidirectional = True
        self.learning_rate = 0.0005
        self.batch_size = 32
        self.epochs = 50
        self.patience = 10

    def create_experiment_dir(self, exp_name: str) -> Path:
        """Create directory for this ablation experiment"""
        exp_dir = self.base_dir / exp_name / self.timestamp
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / 'models').mkdir(exist_ok=True)
        (exp_dir / 'figures').mkdir(exist_ok=True)
        (exp_dir / 'logs').mkdir(exist_ok=True)
        return exp_dir

    def log(self, message: str, log_file: Path):
        """Log to file and console"""
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')

    def load_subject_data(self, subject_id: str):
        """Load subject data"""
        file_path = self.data_dir / f"{subject_id}.csv"
        df = pd.read_csv(file_path)
        exclude_cols = ['time', 'label', 'label_idx']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols].values
        y = df['label_idx'].values
        return X, y, feature_cols

    def compute_class_weights(self, y_train_list):
        """Compute balanced class weights"""
        y_flat = np.concatenate([y.flatten() for y in y_train_list])
        classes = np.unique(y_flat)
        weights = compute_class_weight('balanced', classes=classes, y=y_flat)
        return torch.FloatTensor(weights)

    def build_model(self, model_type: str, cnn_variant: str, 
                    input_size: int, dropout: float) -> nn.Module:
        """Build model with specified architecture and dropout"""
        
        if model_type == "lstm":
            return LSTMClassifier(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_classes=self.num_classes,
                dropout=dropout,
                bidirectional=self.bidirectional
            )

        elif model_type == "cnn_lstm":
            base_kwargs = dict(
                input_size=input_size,
                num_classes=self.num_classes,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=dropout,
                bidirectional=self.bidirectional
            )

            if cnn_variant == "standard":
                return create_cnn_lstm_model(
                    model_type="standard",
                    cnn_channels=[64, 128, 128],
                    kernel_sizes=[5, 5, 3],
                    **base_kwargs
                )
            elif cnn_variant == "deep":
                return create_cnn_lstm_model(
                    model_type="deep",
                    cnn_channels=[64, 128, 256],
                    kernel_sizes=[7, 5, 3],
                    pool_sizes=[2, 2, 1],
                    **base_kwargs
                )
            elif cnn_variant == "light":
                return create_cnn_lstm_model(
                    model_type="light",
                    **base_kwargs
                )
            else:
                raise ValueError(f"Unknown cnn_variant: {cnn_variant}")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def train_epoch(self, model, dataloader, criterion, optimizer):
        """Train one epoch"""
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(
                logits.reshape(-1, self.num_classes),
                y_batch.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, model, dataloader):
        """Evaluate model"""
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                logits = model(X_batch)
                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y_batch.numpy())

        all_preds = np.concatenate([p.flatten() for p in all_preds])
        all_labels = np.concatenate([l.flatten() for l in all_labels])

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return acc, f1, all_preds, all_labels

    def train_single_fold(self, test_subject: str, all_subjects: list,
                          window_size: float, overlap: float, dropout: float,
                          model_type: str, cnn_variant: str,
                          exp_dir: Path, log_file: Path) -> dict:
        """Train one LOSO fold with specific hyperparameters"""
        
        self.log(f"\n{'='*60}", log_file)
        self.log(f"FOLD: {test_subject} | W={window_size}s | O={overlap} | D={dropout}", log_file)
        self.log(f"{'='*60}", log_file)

        train_subjects = [s for s in all_subjects if s != test_subject]

        # Load data
        train_data = {}
        for subj in train_subjects:
            X, y, _ = self.load_subject_data(subj)
            train_data[subj] = {'X': X, 'y': y}
        X_test_raw, y_test_raw, _ = self.load_subject_data(test_subject)

        # Preprocess
        preprocessor = IMUPreprocessor(method="zscore", use_lowpass=False)
        X_train_all = np.vstack([train_data[s]['X'] for s in train_subjects])
        preprocessor.fit(X_train_all, subject_id='global_train')

        # Window with ablated parameters
        windower = WindowGenerator(window_size=window_size, overlap=overlap)

        X_train_list, y_train_list = [], []
        for subj in train_subjects:
            X_norm = preprocessor.transform(train_data[subj]['X'], 'global_train')
            X_win, y_win = windower.create_windows_sequence_labeling(
                X_norm, train_data[subj]['y']
            )
            X_train_list.append(X_win)
            y_train_list.append(y_win)

        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)

        X_test_norm = preprocessor.transform(X_test_raw, 'global_train')
        X_test, y_test = windower.create_windows_sequence_labeling(
            X_test_norm, y_test_raw
        )

        self.log(f"Train: {X_train.shape} | Test: {X_test.shape}", log_file)

        # DataLoaders
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
            batch_size=self.batch_size, shuffle=False
        )

        # Build model with ablated dropout
        input_size = X_train.shape[2]
        model = self.build_model(model_type, cnn_variant, input_size, dropout).to(self.device)
        n_params = sum(p.numel() for p in model.parameters())

        # Training setup
        class_weights = self.compute_class_weights(y_train_list).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )

        # Training loop
        best_val_f1 = 0.0
        patience_counter = 0
        history = {'train_loss': [], 'val_acc': [], 'val_f1': []}

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            val_acc, val_f1, _, _ = self.evaluate(model, test_loader)

            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            scheduler.step(epoch)

            if epoch % 10 == 0:
                self.log(f"Ep {epoch+1:3d}/{self.epochs} | "
                        f"Loss: {train_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}", 
                        log_file)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.log(f"  Early stopping at epoch {epoch+1}", log_file)
                    break

        # Final evaluation
        final_acc, final_f1, preds, labels = self.evaluate(model, test_loader)
        self.log(f"Final — Acc: {final_acc:.4f} | F1: {final_f1:.4f}", log_file)

        return {
            'test_subject': test_subject,
            'accuracy': float(final_acc),
            'f1_macro': float(final_f1),
            'num_train_windows': len(X_train),
            'num_test_windows': len(X_test),
            'n_params': n_params
        }

    def run_ablation_experiment(self, exp_name: str, model_type: str,
                                cnn_variant: str, ablation_params: dict) -> dict:
        """
        Run full LOSO for one ablation configuration.
        
        Args:
            exp_name: Experiment identifier
            model_type: "lstm" or "cnn_lstm"
            cnn_variant: "standard", "deep", or "light"
            ablation_params: Dict with 'window_size', 'overlap', 'dropout'
        
        Returns:
            results: Dict with mean accuracy, F1, and per-fold results
        """
        
        exp_dir = self.create_experiment_dir(exp_name)
        log_file = exp_dir / 'logs' / 'training.txt'

        window_size = ablation_params['window_size']
        overlap = ablation_params['overlap']
        dropout = ablation_params['dropout']

        self.log(f"\n{'='*70}", log_file)
        self.log(f"ABLATION EXPERIMENT: {exp_name}", log_file)
        self.log(f"Model: {model_type} / {cnn_variant}", log_file)
        self.log(f"Window Size: {window_size}s | Overlap: {overlap} | Dropout: {dropout}", log_file)
        self.log(f"{'='*70}", log_file)

        all_subjects = sorted([f.stem for f in self.data_dir.glob('*.csv')])
        self.log(f"Subjects: {all_subjects}", log_file)

        t_start = time.time()
        all_results = []

        for test_subject in all_subjects:
            result = self.train_single_fold(
                test_subject, all_subjects,
                window_size, overlap, dropout,
                model_type, cnn_variant,
                exp_dir, log_file
            )
            all_results.append(result)

        elapsed = time.time() - t_start

        # Aggregate results
        accuracies = [r['accuracy'] for r in all_results]
        f1_scores = [r['f1_macro'] for r in all_results]

        final_results = {
            'exp_name': exp_name,
            'model_type': model_type,
            'cnn_variant': cnn_variant,
            'ablation_params': ablation_params,
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'mean_f1_macro': float(np.mean(f1_scores)),
            'std_f1_macro': float(np.std(f1_scores)),
            'elapsed_minutes': round(elapsed / 60, 2),
            'fold_results': all_results
        }

        # Save results
        results_file = exp_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        self.log(f"\n{'='*70}", log_file)
        self.log(f"EXPERIMENT COMPLETE", log_file)
        self.log(f"Accuracy: {final_results['mean_accuracy']:.4f} ± {final_results['std_accuracy']:.4f}", log_file)
        self.log(f"F1 Macro: {final_results['mean_f1_macro']:.4f} ± {final_results['std_f1_macro']:.4f}", log_file)
        self.log(f"Time: {final_results['elapsed_minutes']:.1f} min", log_file)
        self.log(f"Results saved → {results_file}", log_file)
        self.log(f"{'='*70}", log_file)

        return final_results



class AblationStudyRunner:
    """
    Orchestrates multiple ablation experiments.
    Tests window size, overlap, and dropout systematically.
    """

    def __init__(self, config_path='/kaggle/input/datasets/manelouchar/config/config.yaml'):
        self.trainer = AblationStudyTrainer(config_path)
        self.all_results = []

    def run_window_size_ablation(self, model_type="cnn_lstm", 
                                  cnn_variant="deep"):
        """
        Ablation 1: Window Size
        Test: 1.5s vs 2.0s vs 2.5s
        Fixed: overlap=0.5, dropout=0.2
        """
        print("\n" + "="*70)
        print("ABLATION 1: Window Size")
        print("="*70)

        window_sizes = [1.5, 2.0, 2.5]
        fixed_params = {'overlap': 0.5, 'dropout': 0.2}

        for ws in window_sizes:
            exp_name = f"window_size_{ws:.1f}s"
            params = {**fixed_params, 'window_size': ws}
            
            print(f"\n▶ Running {exp_name}...")
            result = self.trainer.run_ablation_experiment(
                exp_name, model_type, cnn_variant, params
            )
            self.all_results.append(result)
            print(f"✓ {exp_name} complete — Acc: {result['mean_accuracy']:.4f}, "
                  f"F1: {result['mean_f1_macro']:.4f}")

    def run_overlap_ablation(self, model_type="cnn_lstm", 
                            cnn_variant="deep"):
        """
        Ablation 2: Overlap
        Test: 0.25 vs 0.5 vs 0.75
        Fixed: window_size=2.0s, dropout=0.2
        """
        print("\n" + "="*70)
        print("ABLATION 2: Overlap")
        print("="*70)

        overlaps = [0.25, 0.5, 0.75]
        fixed_params = {'window_size': 2.0, 'dropout': 0.2}

        for ovl in overlaps:
            exp_name = f"overlap_{ovl:.2f}"
            params = {**fixed_params, 'overlap': ovl}
            
            print(f"\n▶ Running {exp_name}...")
            result = self.trainer.run_ablation_experiment(
                exp_name, model_type, cnn_variant, params
            )
            self.all_results.append(result)
            print(f"✓ {exp_name} complete — Acc: {result['mean_accuracy']:.4f}, "
                  f"F1: {result['mean_f1_macro']:.4f}")

    def run_dropout_ablation(self, model_type="cnn_lstm", 
                            cnn_variant="deep"):
        """
        Ablation 3: Dropout Rate
        Test: 0.1 vs 0.2 vs 0.3
        Fixed: window_size=2.0s, overlap=0.5
        """
        print("\n" + "="*70)
        print("ABLATION 3: Dropout Rate")
        print("="*70)

        dropouts = [0.1, 0.2, 0.3]
        fixed_params = {'window_size': 2.0, 'overlap': 0.5}

        for dr in dropouts:
            exp_name = f"dropout_{dr:.1f}"
            params = {**fixed_params, 'dropout': dr}
            
            print(f"\n▶ Running {exp_name}...")
            result = self.trainer.run_ablation_experiment(
                exp_name, model_type, cnn_variant, params
            )
            self.all_results.append(result)
            print(f"✓ {exp_name} complete — Acc: {result['mean_accuracy']:.4f}, "
                  f"F1: {result['mean_f1_macro']:.4f}")

    def run_all_ablations(self, model_type="cnn_lstm", cnn_variant="deep"):
        """Run all three ablation studies sequentially"""
        print("\n" + "="*70)
        print(f"FULL ABLATION STUDY: {model_type} / {cnn_variant}")
        print("="*70)

        t_total_start = time.time()

        self.run_window_size_ablation(model_type, cnn_variant)
        self.run_overlap_ablation(model_type, cnn_variant)
        self.run_dropout_ablation(model_type, cnn_variant)

        t_total = time.time() - t_total_start

        print("\n" + "="*70)
        print(f"ALL ABLATIONS COMPLETE — Total time: {t_total/3600:.1f} hours")
        print("="*70)

        self.generate_summary_report()
        self.plot_ablation_results()

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        summary_file = self.trainer.base_dir / f'ablation_summary_{self.trainer.timestamp}.txt'

        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ABLATION STUDY SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total experiments: {len(self.all_results)}\n\n")

            # Group by ablation type
            window_results = [r for r in self.all_results if 'window_size' in r['exp_name']]
            overlap_results = [r for r in self.all_results if 'overlap' in r['exp_name']]
            dropout_results = [r for r in self.all_results if 'dropout' in r['exp_name']]

            for title, results in [
                ("WINDOW SIZE ABLATION", window_results),
                ("OVERLAP ABLATION", overlap_results),
                ("DROPOUT ABLATION", dropout_results)
            ]:
                if not results:
                    continue

                f.write("="*70 + "\n")
                f.write(f"{title}\n")
                f.write("="*70 + "\n\n")

                f.write(f"{'Config':<20} {'Accuracy':>12} {'Std':>8} {'F1 Macro':>10} {'Std':>8} {'Time(min)':>10}\n")
                f.write("-"*70 + "\n")

                for r in results:
                    config_str = r['exp_name'].replace('_', ' ')
                    f.write(f"{config_str:<20} "
                           f"{r['mean_accuracy']:>12.4f} "
                           f"{r['std_accuracy']:>8.4f} "
                           f"{r['mean_f1_macro']:>10.4f} "
                           f"{r['std_f1_macro']:>8.4f} "
                           f"{r['elapsed_minutes']:>10.1f}\n")

                # Best config
                best = max(results, key=lambda x: x['mean_f1_macro'])
                f.write("\n")
                f.write(f"Best: {best['exp_name']} — "
                       f"Acc: {best['mean_accuracy']:.4f}, "
                       f"F1: {best['mean_f1_macro']:.4f}\n\n")

            f.write("="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")

        print(f"\n✓ Summary report → {summary_file}")

    def plot_ablation_results(self):
        """Generate comparison plots for all ablations"""
        
        # Group results
        window_results = [r for r in self.all_results if 'window_size' in r['exp_name']]
        overlap_results = [r for r in self.all_results if 'overlap' in r['exp_name']]
        dropout_results = [r for r in self.all_results if 'dropout' in r['exp_name']]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Window Size
        if window_results:
            ws_vals = [r['ablation_params']['window_size'] for r in window_results]
            ws_accs = [r['mean_accuracy'] for r in window_results]
            ws_f1s = [r['mean_f1_macro'] for r in window_results]

            axes[0].plot(ws_vals, ws_accs, 'o-', label='Accuracy', linewidth=2, markersize=8)
            axes[0].plot(ws_vals, ws_f1s, 's-', label='F1 Macro', linewidth=2, markersize=8)
            axes[0].set_xlabel('Window Size (s)', fontsize=12)
            axes[0].set_ylabel('Score', fontsize=12)
            axes[0].set_title('Window Size Ablation', fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # Plot 2: Overlap
        if overlap_results:
            ovl_vals = [r['ablation_params']['overlap'] for r in overlap_results]
            ovl_accs = [r['mean_accuracy'] for r in overlap_results]
            ovl_f1s = [r['mean_f1_macro'] for r in overlap_results]

            axes[1].plot(ovl_vals, ovl_accs, 'o-', label='Accuracy', linewidth=2, markersize=8)
            axes[1].plot(ovl_vals, ovl_f1s, 's-', label='F1 Macro', linewidth=2, markersize=8)
            axes[1].set_xlabel('Overlap', fontsize=12)
            axes[1].set_ylabel('Score', fontsize=12)
            axes[1].set_title('Overlap Ablation', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # Plot 3: Dropout
        if dropout_results:
            dr_vals = [r['ablation_params']['dropout'] for r in dropout_results]
            dr_accs = [r['mean_accuracy'] for r in dropout_results]
            dr_f1s = [r['mean_f1_macro'] for r in dropout_results]

            axes[2].plot(dr_vals, dr_accs, 'o-', label='Accuracy', linewidth=2, markersize=8)
            axes[2].plot(dr_vals, dr_f1s, 's-', label='F1 Macro', linewidth=2, markersize=8)
            axes[2].set_xlabel('Dropout Rate', fontsize=12)
            axes[2].set_ylabel('Score', fontsize=12)
            axes[2].set_title('Dropout Ablation', fontsize=14, fontweight='bold')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = self.trainer.base_dir / f'ablation_comparison_{self.trainer.timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Ablation plots → {plot_file}")
