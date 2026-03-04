from src.data.preprocessing import IMUPreprocessor
from src.data.windowing import WindowGenerator
from src.models.factory import create_cnn_lstm_model
from src.models.lstm_frame import LSTMClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import yaml
import json
import time
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, f1_score, 
                             classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')



class LOSOTrainer:
    """
    Generic LOSO Trainer for LSTM and CNN-LSTM variants.
    Interface: (batch, seq_len, features) → (batch, seq_len, classes)
    """

    def __init__(
        self,
        model_type: str = "lstm",
        cnn_variant: str = "standard",
        config_path: str = '/kaggle/input/datasets/manelouchar/config/config.yaml'
    ):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_type  = model_type
        self.cnn_variant = cnn_variant

        # ── Shared hyperparameters ──────────────────────────────
        self.window_size   = 2.5
        self.overlap       = 0.75
        self.num_classes   = len(self.config['data']['classes'])
        self.hidden_size   = 256
        self.num_layers    = 2
        self.dropout       = 0.3
        self.bidirectional = True
        self.learning_rate = 0.0005
        self.batch_size    = 32
        self.epochs        = 50
        self.patience      = 10

        # ── CNN-specific ────────────────────────────────────────
        self.cnn_channels = [64, 128, 128]
        self.kernel_sizes = [5, 5, 3]
        self.pool_sizes   = [2, 2, 1]          # deep variant only

        # ── Device & paths ──────────────────────────────────────
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir  = Path("/kaggle/input/datasets/manelouchar/processed")

        variant_tag = (f"{model_type}_{cnn_variant}" 
                       if model_type == "cnn_lstm" else model_type)
        base = Path('/kaggle/working')
        self.models_dir  = base / f'models_{variant_tag}'
        self.logs_dir    = base / f'logs_{variant_tag}'
        self.figures_dir = base / f'figures_{variant_tag}'
        self.reports_dir = base / f'reports_{variant_tag}'

        for d in [self.models_dir, self.logs_dir,
                  self.figures_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.log_file = self.logs_dir / f'{variant_tag}_log_{self.timestamp}.txt'

        self.log("=" * 70)
        self.log(f"Trainer: {variant_tag} | Device: {self.device}")
        self.log(f"Window: {self.window_size}s | Overlap: {self.overlap} | "
                 f"LR: {self.learning_rate} | Epochfs: {self.epochs}")
        self.log("=" * 70)

    # ── Logging ─────────────────────────────────────────────────────────
    def log(self, message: str):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    # ── Data loading ─────────────────────────────────────────────────────
    def load_subject_data(self, subject_id: str):
        file_path = self.data_dir / f"{subject_id}.csv"
        df = pd.read_csv(file_path)

        exclude_cols = ['time', 'label', 'label_idx']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].values
        y = df['label_idx'].values
        return X, y, feature_cols

    # ── Class weights ────────────────────────────────────────────────────
    def compute_class_weights(self, y_train_list):
        y_flat   = np.concatenate([y.flatten() for y in y_train_list])
        classes  = np.unique(y_flat)
        weights  = compute_class_weight('balanced', classes=classes, y=y_flat)
        return torch.FloatTensor(weights)

    # ── Model factory ────────────────────────────────────────────────────
    def _build_model(self, input_size: int) -> nn.Module:
        
        if self.model_type == "lstm":
            return LSTMClassifier(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_classes=self.num_classes,
                dropout=self.dropout,
                bidirectional=self.bidirectional
            )

        elif self.model_type == "cnn_lstm":
            base_kwargs = dict(
                input_size=input_size,
                num_classes=self.num_classes,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional
            )

            if self.cnn_variant == "standard":
                return create_cnn_lstm_model(
                    model_type="standard",
                    cnn_channels=self.cnn_channels,
                    kernel_sizes=self.kernel_sizes,
                    **base_kwargs
                )

            elif self.cnn_variant == "deep":
                return create_cnn_lstm_model(
                    model_type="deep",
                    cnn_channels=[64, 128, 256],
                    kernel_sizes=[7, 5, 3],
                    pool_sizes=self.pool_sizes,
                    **base_kwargs
                )

            elif self.cnn_variant == "light":
                # Factory internally overrides hidden_size=64, num_layers=1
                return create_cnn_lstm_model(
                    model_type="light",
                    **base_kwargs
                )

            else:
                raise ValueError(f"Unknown cnn_variant: {self.cnn_variant}")

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    # ── Training ─────────────────────────────────────────────────────────
    def train_epoch(self, model, dataloader, criterion, optimizer) -> float:
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            logits = model(X_batch)                          # (B, T, C)

            loss = criterion(
                logits.reshape(-1, self.num_classes),        # (B*T, C)
                y_batch.reshape(-1)                          # (B*T,)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    # ── Evaluation ───────────────────────────────────────────────────────
    def evaluate(self, model, dataloader):
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                logits  = model(X_batch)
                preds   = torch.argmax(logits, dim=-1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(y_batch.numpy())

        all_preds  = np.concatenate([p.flatten() for p in all_preds])
        all_labels = np.concatenate([l.flatten() for l in all_labels])

        acc      = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return acc, f1_macro, all_preds, all_labels

    # ── Plotting ─────────────────────────────────────────────────────────
    def plot_training_curves(self, history: dict, test_subject: str):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(history['train_loss']) + 1)

        axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2)
        axes[0].set_title(f'Loss — {test_subject}')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history['val_acc'], 'g-', label='Accuracy', linewidth=2)
        axes[1].plot(epochs, history['val_f1'],  'r-', label='F1 Macro', linewidth=2)
        axes[1].set_title(f'Validation — {test_subject}')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Score')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / f'curves_{test_subject}.png', dpi=150)
        plt.close()

    def plot_confusion_matrix(self, cm: np.ndarray, test_subject: str):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=self.config['data']['classes'],
            yticklabels=self.config['data']['classes']
        )
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix — {test_subject}')
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'cm_{test_subject}.png', dpi=150)
        plt.close()

    def plot_summary_results(self, all_results: list):
        subjects   = [r['test_subject'] for r in all_results]
        accuracies = [r['accuracy']     for r in all_results]
        f1_scores  = [r['f1_macro']     for r in all_results]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        mean_acc = np.mean(accuracies)
        mean_f1  = np.mean(f1_scores)

        for ax, values, mean, title, ylabel in [
            (axes[0], accuracies, mean_acc, 'Accuracy (LOSO)', 'Accuracy'),
            (axes[1], f1_scores,  mean_f1,  'F1 Macro (LOSO)', 'F1 Macro'),
        ]:
            colors = ['green' if v > mean else 'coral' for v in values]
            ax.bar(subjects, values, color=colors, alpha=0.7, edgecolor='black')
            ax.axhline(mean, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean:.4f}')
            ax.set_title(f'{self.model_type} {self.cnn_variant} — {title}')
            ax.set_xlabel('Test Subject'); ax.set_ylabel(ylabel)
            ax.legend(); ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'summary.png', dpi=300)
        plt.close()

    # ── Per-fold training ────────────────────────────────────────────────
    def train_fold(self, test_subject: str, all_subjects: list) -> dict:
        self.log(f"\n{'='*70}")
        self.log(f"FOLD: {test_subject} | {self.model_type} / {self.cnn_variant}")
        self.log(f"{'='*70}")

        train_subjects = [s for s in all_subjects if s != test_subject]
        val_subject    = train_subjects[-1]          # last subject as validation
        train_subjects = train_subjects[:-1]         # remaining 8 for training

        # Load
        train_data = {}
        for subj in train_subjects:
            X, y, _ = self.load_subject_data(subj)
            train_data[subj] = {'X': X, 'y': y}
        X_test_raw, y_test_raw, _ = self.load_subject_data(test_subject)

        # Preprocess — fit on train only
        preprocessor = IMUPreprocessor(method="zscore", use_lowpass=False)
        X_train_all  = np.vstack([train_data[s]['X'] for s in train_subjects])
        preprocessor.fit(X_train_all, subject_id='global_train')

        # Window
        windower = WindowGenerator(window_size=self.window_size, overlap=self.overlap)

        # ── load val subject raw data ──
        X_val_raw, y_val_raw, _ = self.load_subject_data(val_subject)
        
        # ── fit preprocessor on train subjects only (exclude val too) ──
        X_train_all  = np.vstack([train_data[s]['X'] for s in train_subjects])
        preprocessor.fit(X_train_all, subject_id='global_train')
        
        X_train_list, y_train_list = [], []
        for subj in train_subjects:
            X_norm       = preprocessor.transform(train_data[subj]['X'], 'global_train')
            X_win, y_win = windower.create_windows_sequence_labeling(
                X_norm, train_data[subj]['y']
            )
            X_train_list.append(X_win)
            y_train_list.append(y_win)
        
        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        # ── validation set (held-out train subject, never the test subject) ──
        X_val_norm     = preprocessor.transform(X_val_raw, 'global_train')
        X_val, y_val   = windower.create_windows_sequence_labeling(X_val_norm, y_val_raw)
        
        # ── true test set (only touched for final evaluation) ──
        X_test_norm    = preprocessor.transform(X_test_raw, 'global_train')
        X_test, y_test = windower.create_windows_sequence_labeling(X_test_norm, y_test_raw)

        self.log(f"Train subjects: {train_subjects} | Val: {val_subject} | Test: {test_subject}")
        self.log(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

        # DataLoaders
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=self.batch_size, shuffle=False
        )

        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
            batch_size=self.batch_size, shuffle=False
        )

        # Model
        input_size = X_train.shape[2]
        model      = self._build_model(input_size).to(self.device)
        n_params   = sum(p.numel() for p in model.parameters())
        self.log(f"Parameters: {n_params:,}")

        # Loss / optimizer / scheduler
        class_weights = self.compute_class_weights(y_train_list).to(self.device)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)
        optimizer     = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler     = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )

        # Training loop
        best_val_f1     = 0.0
        patience_counter = 0
        history         = {'train_loss': [], 'val_acc': [], 'val_f1': []}
        model_path      = self.models_dir / f'{self.model_type}_{self.cnn_variant}_{test_subject}.pth'

        for epoch in range(self.epochs):
            train_loss         = self.train_epoch(model, train_loader, criterion, optimizer)
            val_acc, val_f1, _, _ = self.evaluate(model, val_loader)    

            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            scheduler.step(epoch)

            self.log(f"Ep {epoch+1:3d}/{self.epochs} | "
                     f"Loss: {train_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1      = val_f1
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_f1':           val_f1,
                    'epoch':            epoch,
                    'n_params':         n_params
                }, model_path)
                self.log(f"  ✓ Best saved (F1: {val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.log(f"  Early stopping at epoch {epoch+1}")
                    break

        # Final evaluation with best checkpoint
        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        final_acc, final_f1, preds, labels = self.evaluate(model, test_loader)

        self.log(f"\nFinal — Acc: {final_acc:.4f} | F1: {final_f1:.4f}")

        self.plot_training_curves(history, test_subject)
        cm     = confusion_matrix(labels, preds)
        self.plot_confusion_matrix(cm, test_subject)

        report = classification_report(
            labels, preds,
            target_names=self.config['data']['classes'],
            output_dict=True
        )
        self.log(classification_report(
            labels, preds, target_names=self.config['data']['classes']
        ))

        return {
            'test_subject':      test_subject,
            'accuracy':          float(final_acc),
            'f1_macro':          float(final_f1),
            'per_class_f1':      {cls: report[cls]['f1-score']
                                  for cls in self.config['data']['classes']},
            'num_train_windows': int(len(X_train)),
            'num_test_windows':  int(len(X_test)),
            'n_params':          n_params,
            'confusion_matrix':  cm.tolist(),
            'training_history':  history
        }

    # ── LOSO loop ────────────────────────────────────────────────────────
    def run_loso(self) -> dict:
        all_files    = sorted(self.data_dir.glob('*.csv'))
        all_subjects = [f.stem for f in all_files]

        self.log(f"\nSubjects ({len(all_subjects)}): {all_subjects}")

        all_results = []
        for test_subject in all_subjects:
            result = self.train_fold(test_subject, all_subjects)
            all_results.append(result)

        accuracies = [r['accuracy'] for r in all_results]
        f1_scores  = [r['f1_macro'] for r in all_results]

        self.log(f"\n{'='*70}")
        self.log("LOSO COMPLETE")
        self.log(f"Accuracy : {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        self.log(f"F1 Macro : {np.mean(f1_scores):.4f}  ± {np.std(f1_scores):.4f}")
        self.log(f"{'='*70}")

        self.plot_summary_results(all_results)

        # Save JSON
        final = {
            'timestamp':      self.timestamp,
            'model_type':     self.model_type,
            'cnn_variant':    self.cnn_variant,
            'mean_accuracy':  float(np.mean(accuracies)),
            'std_accuracy':   float(np.std(accuracies)),
            'mean_f1_macro':  float(np.mean(f1_scores)),
            'std_f1_macro':   float(np.std(f1_scores)),
            'fold_results':   all_results
        }

        out = self.models_dir / f'results_{self.timestamp}.json'
        with open(out, 'w') as f:
            json.dump(final, f, indent=2)
        self.log(f"✓ Results → {out}")

        return final

