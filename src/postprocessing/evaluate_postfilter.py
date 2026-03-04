import datetime
import numpy as np
from sklearn.base import accuracy_score
from sklearn.metrics import f1_score
from streamlit import json
import torch
from pathlib import Path
import torch.nn as nn
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from src.data.preprocessing import IMUPreprocessor
from src.data.windowing import WindowGenerator
from src.models.cnn_lstm import CNNLSTMClassifier, CNNLSTMClassifierDeep
from src.models.lstm_frame import LSTMClassifier
from src.postprocessing.dmin_sweeper import DMinSweeper
from src.postprocessing.duration_filter import DurationFilter
from src.postprocessing.transition_mask import STATE2IDX
from src.postprocessing.transition_mask import STATES
from src.postprocessing.viterbi import ViterbiDecoder

class ModelLoader:
    _DEFAULTS = {
        'lstm': dict(
            hidden_size=256, num_layers=2, dropout=0.2, bidirectional=False
        ),
        'cnn_lstm': dict(
            cnn_channels=[64, 128, 128], kernel_sizes=[5, 5, 3],
            hidden_size=256, num_layers=2, dropout=0.2, bidirectional=True
        ),
        'cnn_lstm_deep': dict(
            cnn_channels=[64, 128, 256], kernel_sizes=[7, 5, 3],
            pool_sizes=[2, 2, 1],
            hidden_size=256, num_layers=2, dropout=0.2, bidirectional=True
        ),
    }

    def __init__(self, models_dir: Path, model_type: str, cfg: dict):
        self.models_dir = models_dir
        self.model_type = model_type
        self.overrides  = cfg.get('model', {})

    def load(self, test_subject: str, input_size: int,
             num_classes: int) -> nn.Module:
        ckpt = self.models_dir / f'cnn_lstm_deep_{test_subject}.pth'
        if not ckpt.exists():
            ckpt = self.models_dir / f'model_{self.model_type}_{test_subject}.pth'
        print(f"  Loading: {ckpt.name}")
        state = torch.load(ckpt, weights_only=False, map_location='cpu')
        model = self._build(input_size, num_classes)
        model.load_state_dict(state['model_state_dict'])
        model.eval()
        return model

    def _build(self, input_size: int, num_classes: int) -> nn.Module:
        kw = {**self._DEFAULTS[self.model_type], **self.overrides}
        if self.model_type == 'lstm':
            return LSTMClassifier(input_size, num_classes=num_classes, **kw)
        if self.model_type == 'cnn_lstm':
            return CNNLSTMClassifier(input_size, num_classes=num_classes, **kw)
        if self.model_type == 'cnn_lstm_deep':
            return CNNLSTMClassifierDeep(input_size, num_classes=num_classes, **kw)
        raise ValueError(f"Unknown model_type: {self.model_type}")

class Inferencer:
    def __init__(self, batch_size: int = 64):
        self.bs = batch_size

    def get_logits(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: (N, T, F)
        Returns:
            logits: (N*T, C) — flattened frame-wise logits
        """
        chunks = []
        with torch.no_grad():
            for i in range(0, len(X), self.bs):
                batch = torch.FloatTensor(X[i:i+self.bs])
                chunks.append(model(batch).numpy())   # (b, T, C)
        logits = np.concatenate(chunks, axis=0)       # (N, T, C)
        return logits.reshape(-1, logits.shape[-1])   # (N*T, C)



class PostFilterEvaluator:
    # ── 5 METHODS ────────────────────────────────────────────────────
    METHODS = ['raw', 'viterbi', 'vit_dur', 'vit_hardcoded', 'vit_best']
    LABELS  = {
        'raw':            'Raw argmax',
        'viterbi':        'Viterbi',
        'vit_dur':        'Viterbi + d_min (estimated Q25)',
        'vit_hardcoded':  'Viterbi + d_min (hardcoded EDA Q1)',  # ← NEW
        'vit_best':       'Viterbi + d_min (best swept)',
    }

    def __init__(self, config_path: str, model_type: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.model_type  = model_type
        self.num_classes = len(self.cfg["data"]["classes"])
        self.window_size = self.cfg["data"]["window_size"]
        self.overlap = self.cfg["data"]["overlap"]
        self.sr = self.cfg["data"]["sampling_rate"]

        self.data_dir = Path(self.cfg["paths"]["processed_dir"])
        self.models_dir = Path(self.cfg["paths"]["models_dir"])
        self.results_dir = Path(self.cfg["paths"]["results_dir"])
        self.figures_dir = self.results_dir / "figures"

        self.timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.loader     = ModelLoader(self.models_dir, model_type, self.cfg)
        self.inferencer = Inferencer()
        self.viterbi    = ViterbiDecoder()
        self.sweeper    = DMinSweeper(range(1, 121, 5), self.sr)

        # ── Hardcoded d_min from EDA (convert seconds → frames) ──
        self.dmin_hardcoded = {
            STATE2IDX[state]: int(sec * self.sr)
            for state, sec in self.cfg["postfilter"]["hardcoded_dmin_sec"].items()
        }
        print(f"\nHardcoded d_min (frames): {self.dmin_hardcoded}")

    # ── Data ─────────────────────────────────────────────────

    def _load_subject(self, sid: str):
        df = pd.read_csv(self.data_dir / f'{sid}.csv')
        ex = ['time', 'label', 'label_idx']
        fc = [c for c in df.columns if c not in ex]
        return df[fc].values, df['label_idx'].values

    def _build_test_windows(self, all_subjects: list, test_subject: str):
        train_subjects = [s for s in all_subjects if s != test_subject]

        train_data = {}
        for s in train_subjects:
            X, y = self._load_subject(s)
            train_data[s] = {'X': X, 'y': y}
        X_test_raw, y_test_raw = self._load_subject(test_subject)

        pre = IMUPreprocessor(method='zscore', use_lowpass=True,
                              cutoff_hz=25.0, sampling_rate=self.sr)
        X_train_all = np.vstack([train_data[s]['X'] for s in train_subjects])
        pre.fit(X_train_all, subject_id='global')

        win    = WindowGenerator(window_size=self.window_size, overlap=self.overlap)
        X_norm = pre.transform(X_test_raw, subject_id='global')
        X_win, y_win = win.create_windows_sequence_labeling(X_norm, y_test_raw)

        y_train_all = np.concatenate([train_data[s]['y'] for s in train_subjects])
        return X_win, y_win, y_train_all

    # ── Single fold ──────────────────────────────────────────

    def _evaluate_fold(self, test_subject: str, all_subjects: list) -> dict:
        print(f"\n{'='*60}")
        print(f"FOLD: {test_subject}")
        print(f"{'='*60}")

        X_win, y_win, y_train_all = self._build_test_windows(
            all_subjects, test_subject)

        d_min_est = DurationFilter.estimate_from_labels(
            y_train_all, self.num_classes,
            quantile=0.05, sampling_rate=self.sr)

        model  = self.loader.load(test_subject, X_win.shape[2], self.num_classes)
        logits = self.inferencer.get_logits(model, X_win)
        gt     = y_win.flatten()

        # 5 decoding methods
        raw_preds = ViterbiDecoder.argmax_baseline(logits)
        vit_preds = self.viterbi.decode(logits)

        filt_est = DurationFilter(d_min_est, self.sr)
        vit_dur  = filt_est.apply(vit_preds)

        # ── NEW: hardcoded d_min from EDA ────────────────────────
        filt_hardcoded = DurationFilter(self.dmin_hardcoded, self.sr)
        vit_hardcoded  = filt_hardcoded.apply(vit_preds)

        sweep    = self.sweeper.sweep(vit_preds, gt)
        self.sweeper.plot(sweep, test_subject, self.figures_dir)

        filt_best = DurationFilter(
            {c: sweep['best_dmin'] for c in range(self.num_classes)}, self.sr)
        vit_best  = filt_best.apply(vit_preds)

        print(f"  Best d_min (swept) : {sweep['best_dmin']} frames "
              f"({sweep['best_dmin']/self.sr*1000:.0f} ms)")

        def _metrics(preds, label):
            acc = accuracy_score(gt, preds)
            f1  = f1_score(gt, preds, average='macro', zero_division=0)
            fc  = filt_est.false_changes_per_min(preds, gt)
            lat = filt_est.transition_latency(preds, gt)
            print(f"  [{label:36s}]  Acc={acc:.4f}  F1={f1:.4f}  "
                  f"FC/min={fc:.2f}  Lat={lat['median_ms']:.1f}ms")
            return {'accuracy': acc, 'f1_macro': f1, 'fc_per_min': fc, **lat}

        return {
            'test_subject':      test_subject,
            'd_min_estimated':   d_min_est,
            'd_min_hardcoded':   self.dmin_hardcoded,
            'sweep':             sweep,
            'raw':               _metrics(raw_preds,      'Raw argmax'),
            'viterbi':           _metrics(vit_preds,      'Viterbi'),
            'vit_dur':           _metrics(vit_dur,        'Viterbi+d_min(est Q25)'),
            'vit_hardcoded':     _metrics(vit_hardcoded,  'Viterbi+d_min(hardcoded EDA Q1)'),
            'vit_best':          _metrics(vit_best,       'Viterbi+d_min(best swept)'),
        }

    # ── Full LOSO run ────────────────────────────────────────

    def run(self) -> list:
        all_subjects = sorted(f.stem for f in self.data_dir.glob('*.csv'))
        print(f"Subjects : {all_subjects}")
        print(f"Model    : {self.model_type.upper()}")

        all_results = []
        for subj in all_subjects:
            all_results.append(self._evaluate_fold(subj, all_subjects))

        self._print_final_summary(all_results)
        self._save_txt_report(all_results)

        json_path = self.results_dir / \
            f'postfilter_{self.model_type}_{self.timestamp}.json'
        with open(json_path, 'w') as fh:
            json.dump(all_results, fh, indent=2, default=str)

        print(f"\n✓ JSON       → {json_path}")
        return all_results

    # ── Console summary ──────────────────────────────────────

    def _print_final_summary(self, all_results: list) -> None:
        def _mean(key, sub):
            vals = [r[key][sub] for r in all_results
                    if not np.isnan(r[key].get(sub, np.nan))]
            return float(np.mean(vals)) if vals else np.nan

        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY — {self.model_type.upper()}")
        print(f"{'='*80}")
        print(f"  {'Method':<40} {'Accuracy':>10} {'F1':>8} "
              f"{'FC/min':>8} {'Lat(ms)':>9}")
        print("  " + "-" * 78)
        for m in self.METHODS:
            print(f"  {self.LABELS[m]:<40} "
                  f"{_mean(m,'accuracy'):>10.4f} "
                  f"{_mean(m,'f1_macro'):>8.4f} "
                  f"{_mean(m,'fc_per_min'):>8.2f} "
                  f"{_mean(m,'median_ms'):>9.1f}")

        fc_raw        = _mean('raw',           'fc_per_min')
        fc_hardcoded  = _mean('vit_hardcoded', 'fc_per_min')
        fc_best       = _mean('vit_best',      'fc_per_min')
        acc_raw       = _mean('raw',           'accuracy')
        acc_hardcoded = _mean('vit_hardcoded', 'accuracy')
        acc_best      = _mean('vit_best',      'accuracy')

        print(f"\n  Hardcoded vs Raw:")
        if fc_raw > 0:
            print(f"    FC/min reduction : {(fc_raw-fc_hardcoded)/fc_raw*100:.1f}%")
        print(f"    Accuracy delta   : {acc_hardcoded-acc_raw:+.4f}")

        print(f"\n  Best Swept vs Raw:")
        if fc_raw > 0:
            print(f"    FC/min reduction : {(fc_raw-fc_best)/fc_raw*100:.1f}%")
        print(f"    Accuracy delta   : {acc_best-acc_raw:+.4f}")
        print(f"{'='*80}")

    # ── TXT report ───────────────────────────────────────────

    def _save_txt_report(self, all_results: list) -> None:
        sep      = "=" * 80
        txt_path = self.results_dir / \
            f'postfilter_report_{self.model_type}_{self.timestamp}.txt'

        def _mean(key, sub):
            vals = [r[key][sub] for r in all_results
                    if not np.isnan(r[key].get(sub, np.nan))]
            return float(np.mean(vals)) if vals else np.nan

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(sep + "\n")
            f.write(f"POST-FILTER EVALUATION REPORT — {self.model_type.upper()}\n")
            f.write(sep + "\n\n")
            f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model     : {self.model_type.upper()}\n")
            f.write(f"Subjects  : {len(all_results)}\n\n")

            # Hardcoded d_min values
            f.write(sep + "\n")
            f.write("HARDCODED d_min VALUES (from EDA Q1)\n")
            f.write(sep + "\n\n")
            for state, sec in self.cfg["postfilter"]["hardcoded_dmin_sec"].items():
                fr = int(sec * self.sr)
                f.write(f"  {state}: {sec:.2f}s ({fr} frames)\n")
            f.write("\n")

            # Summary table
            f.write(sep + "\n")
            f.write("SUMMARY — mean over all LOSO folds\n")
            f.write(sep + "\n\n")
            f.write(f"  {'Method':<40} {'Accuracy':>10} {'F1 Macro':>10}"
                    f" {'FC/min':>8} {'Lat (ms)':>10}\n")
            f.write("  " + "-" * 80 + "\n")
            for m in self.METHODS:
                f.write(f"  {self.LABELS[m]:<40}"
                        f" {_mean(m,'accuracy'):>10.4f}"
                        f" {_mean(m,'f1_macro'):>10.4f}"
                        f" {_mean(m,'fc_per_min'):>8.2f}"
                        f" {_mean(m,'median_ms'):>10.1f}\n")
            f.write("\n")

            fc_raw        = _mean('raw',           'fc_per_min')
            fc_hardcoded  = _mean('vit_hardcoded', 'fc_per_min')
            fc_best       = _mean('vit_best',      'fc_per_min')
            acc_raw       = _mean('raw',           'accuracy')
            acc_hardcoded = _mean('vit_hardcoded', 'accuracy')
            acc_best      = _mean('vit_best',      'accuracy')

            f.write(f"  Hardcoded EDA Q1 vs Raw:\n")
            if fc_raw > 0:
                f.write(f"    FC/min reduction : {(fc_raw-fc_hardcoded)/fc_raw*100:.1f}%\n")
            f.write(f"    Accuracy delta   : {acc_hardcoded-acc_raw:+.4f}\n\n")

            f.write(f"  Best Swept vs Raw:\n")
            if fc_raw > 0:
                f.write(f"    FC/min reduction : {(fc_raw-fc_best)/fc_raw*100:.1f}%\n")
            f.write(f"    Accuracy delta   : {acc_best-acc_raw:+.4f}\n\n")

            # Per-subject breakdown
            f.write(sep + "\n")
            f.write("PER-SUBJECT BREAKDOWN\n")
            f.write(sep + "\n\n")
            for r in all_results:
                bd = r['sweep']['best_dmin']
                f.write(f"  Subject : {r['test_subject']}   "
                        f"best d_min = {bd} frames ({bd/self.sr*1000:.0f} ms)\n")
                f.write(f"  {'Method':<40} {'Accuracy':>10} {'F1':>8}"
                        f" {'FC/min':>8} {'Lat(ms)':>10}\n")
                f.write("  " + "-" * 78 + "\n")
                for m in self.METHODS:
                    rd  = r[m]
                    lat = f"{rd['median_ms']:.1f}" \
                          if not np.isnan(rd['median_ms']) else "N/A"
                    f.write(f"  {self.LABELS[m]:<40}"
                            f" {rd['accuracy']:>10.4f}"
                            f" {rd['f1_macro']:>8.4f}"
                            f" {rd['fc_per_min']:>8.2f}"
                            f" {lat:>10}\n")
                f.write("\n")

            # d_min comparison table
            f.write(sep + "\n")
            f.write("d_min COMPARISON: ESTIMATED vs HARDCODED (frames | ms)\n")
            f.write(sep + "\n\n")
            f.write(f"  {'Subject':<12}  {'Method':<12}" +
                    "".join(f"  {s:>14}" for s in STATES) + "\n")
            f.write("  " + "-" * (26 + 16 * self.num_classes) + "\n")
            for r in all_results:
                dm_est = r['d_min_estimated']
                dm_hc  = r['d_min_hardcoded']

                row_est = f"  {r['test_subject']:<12}  {'Estimated':<12}"
                for c in range(self.num_classes):
                    fr  = dm_est.get(c, 1)
                    row_est += f"  {fr:>4}fr ({fr/self.sr*1000:>3.0f}ms)"
                f.write(row_est + "\n")

                row_hc = f"  {'':<12}  {'Hardcoded':<12}"
                for c in range(self.num_classes):
                    fr  = dm_hc.get(c, 1)
                    row_hc += f"  {fr:>4}fr ({fr/self.sr*1000:>3.0f}ms)"
                f.write(row_hc + "\n\n")

            f.write(sep + "\n")
            f.write("END OF REPORT\n")
            f.write(sep + "\n")

        print(f"✓ TXT report → {txt_path}")

