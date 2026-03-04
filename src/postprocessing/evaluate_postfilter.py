import datetime
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json
import torch
from pathlib import Path
import torch.nn as nn
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from itertools import product

from src.data.preprocessing import IMUPreprocessor
from src.data.windowing import WindowGenerator
from src.models.cnn_lstm import CNNLSTMClassifier, CNNLSTMClassifierDeep
from src.models.lstm_frame import LSTMClassifier
from src.postprocessing.dmin_sweeper import DMinSweeper, DMinSweeperPerState
from src.postprocessing.duration_filter import DurationFilter
from src.postprocessing.transition_mask import STATE2IDX, STATES
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
                batch = torch.FloatTensor(X[i:i + self.bs])
                chunks.append(model(batch).numpy())   # (b, T, C)
        logits = np.concatenate(chunks, axis=0)       # (N, T, C)
        return logits.reshape(-1, logits.shape[-1])   # (N*T, C)


class PostFilterEvaluator:
    """
    Full LOSO post-filter evaluation.

    Methods evaluated:
      raw          : frame-wise argmax (no post-processing)
      viterbi      : constrained Viterbi (hard transition constraint only)
      vit_dur      : Viterbi + estimated d_min (Q10 from training labels)
      vit_hardcoded: Viterbi + hardcoded d_min (EDA-derived Q10)
      vit_uniform  : Viterbi + uniform sweep on validation subject
      vit_perstate : Viterbi + per-state sweep on validation subject

    Strict LOSO separation:
      - d_min estimation uses ONLY training subjects' labels
      - d_min sweep / grid search uses ONLY the validation subject
      - Test subject never touches any tuning step
    """
    METHODS = ['raw', 'viterbi', 'vit_dur', 'vit_hardcoded',
               'vit_uniform', 'vit_perstate']
    LABELS  = {
        'raw':            'Raw argmax',
        'viterbi':        'Viterbi only',
        'vit_dur':        'Vit + d_min (Q10 estimated)',
        'vit_hardcoded':  'Vit + d_min (Q10 hardcoded)',
        'vit_uniform':    'Vit + d_min (uniform sweep)',
        'vit_perstate':   'Vit + d_min (per-state opt.)',
    }

    def __init__(self, config_path: str, model_type: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.model_type  = model_type
        self.num_classes = len(self.cfg['data']['classes'])
        self.sr          = self.cfg['data']['sampling_rate']
        self.window_size = self.cfg['data']['window_size']
        self.overlap     = self.cfg['data']['overlap']

        self.data_dir    = Path(self.cfg['paths']['processed_dir'])
        self.models_dir  = Path(self.cfg['paths']['models_dir'])
        self.results_dir = Path(self.cfg['paths']['results_dir'])
        self.figures_dir = self.results_dir / 'figures'

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        self.loader      = ModelLoader(self.models_dir, model_type, self.cfg)
        self.inferencer  = Inferencer()
        self.viterbi     = ViterbiDecoder()

        # Uniform sweep range: fine-grained near low values, coarser at high
        self.uniform_sweeper = DMinSweeper(
            list(range(3, 31, 2)) + list(range(30, 61, 5)),
            self.sr
        )
        # Per-state ranges: SA/RA/RD transitions develop over longer timescales
        self.perstate_sweeper = DMinSweeperPerState(
            dmin_ranges={
                STATE2IDX['LW']: list(range(5, 16, 2)),
                STATE2IDX['SA']: list(range(8, 21, 2)),
                STATE2IDX['SD']: list(range(5, 16, 2)),
                STATE2IDX['RA']: list(range(8, 21, 2)),
                STATE2IDX['RD']: list(range(8, 21, 2)),
            },
            sampling_rate=self.sr
        )

        # Hardcoded d_min from EDA (Q10 of per-class run-length distributions)
        self.dmin_hardcoded = {
            STATE2IDX[state]: int(sec * self.sr)
            for state, sec in self.cfg['postfilter']['hardcoded_dmin_sec'].items()
        }
        print(f"\nHardcoded d_min (frames): {self.dmin_hardcoded}")

    # ── Data helpers ─────────────────────────────────────────

    def _load_subject(self, sid: str) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(self.data_dir / f'{sid}.csv')
        feat_cols = [c for c in df.columns
                     if c not in ('time', 'label', 'label_idx')]
        return df[feat_cols].values, df['label_idx'].values

    def _build_splits(self, all_subjects: List[str],
                      test_subject: str
                      ) -> Tuple[np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray,
                                 np.ndarray, List[int]]:
        """
        Build test windows, validation windows, and return training
        labels with subject boundary indices for run-length estimation.

        Split:
          train subjects : all except test and val (8 subjects)
          val   subject  : first remaining subject after removing test
          test  subject  : held-out fold subject
        """
        remaining      = [s for s in all_subjects if s != test_subject]
        val_subject    = remaining[0]
        train_subjects = remaining[1:]

        print(f"  Train : {train_subjects}")
        print(f"  Val   : {val_subject}")
        print(f"  Test  : {test_subject}")

        # Fit scaler on training data only (no leakage)
        train_arrays = [self._load_subject(s) for s in train_subjects]
        X_train_all  = np.vstack([a[0] for a in train_arrays])

        pre = IMUPreprocessor(sampling_rate=self.sr)
        pre.fit(X_train_all, 'global')

        win = WindowGenerator(self.window_size, self.overlap, self.sr)

        def _windows(sid):
            Xr, yr = self._load_subject(sid)
            Xn     = pre.transform(Xr, 'global')
            return win.create_windows(Xn, yr)

        X_test, y_test = _windows(test_subject)
        X_val,  y_val  = _windows(val_subject)

        # Training labels with boundary indices for boundary-aware
        # run-length estimation (prevents artefactual short runs at
        # subject concatenation points)
        y_parts    = [a[1] for a in train_arrays]
        boundaries = []
        running    = 0
        for yp in y_parts[:-1]:
            running += len(yp)
            boundaries.append(running)
        y_train_all = np.concatenate(y_parts)

        return X_test, y_test, X_val, y_val, y_train_all, boundaries

    # ── Metrics helper ────────────────────────────────────────

    def _compute_metrics(self,
                         preds: np.ndarray,
                         gt: np.ndarray,
                         filt: DurationFilter,
                         label: str) -> dict:
        acc = accuracy_score(gt, preds)
        f1  = f1_score(gt, preds, average='macro', zero_division=0)
        fc  = filt.false_changes_per_min(preds, gt)
        lat = filt.transition_latency(preds, gt)
        print(f"  [{label:42s}]  Acc={acc:.4f}  F1={f1:.4f}  "
              f"FC/min={fc:.2f}  Lat|med|={lat['median_abs_ms']:.1f}ms  "
              f"Det={lat['detection_rate']*100:.1f}%")
        return dict(accuracy=acc, f1_macro=f1, fc_per_min=fc, **lat)

    # ── Single LOSO fold ─────────────────────────────────────

    def _evaluate_fold(self,
                       test_subject: str,
                       all_subjects: List[str]) -> dict:
        print(f"\n{'='*62}")
        print(f"  FOLD : {test_subject}")
        print(f"{'='*62}")

        (X_test, y_test,
         X_val,  y_val,
         y_train, boundaries) = self._build_splits(all_subjects, test_subject)

        gt_test = y_test.flatten()   # ground truth — test (used ONLY in metrics)
        gt_val  = y_val.flatten()    # ground truth — val  (used for tuning only)

        # d_min estimated from training labels (boundary-aware)
        d_min_est = DurationFilter.estimate_from_labels(
            y_train, boundaries, self.num_classes,
            quantile=0.10, sampling_rate=self.sr
        )

        # Load model and run inference on both test and val splits
        model       = self.loader.load(test_subject, X_test.shape[2], self.num_classes)
        logits_test = self.inferencer.get_logits(model, X_test)
        logits_val  = self.inferencer.get_logits(model, X_val)

        # ── Baseline predictions ──────────────────────────────
        raw_preds = ViterbiDecoder.argmax_baseline(logits_test)
        vit_preds = self.viterbi.decode(logits_test)

        # Validation split predictions (for d_min tuning — never touches test)
        raw_preds_val = ViterbiDecoder.argmax_baseline(logits_val)
        vit_preds_val = self.viterbi.decode(logits_val)
        raw_acc_val   = accuracy_score(gt_val, raw_preds_val)

        # ── d_min tuning on validation split ─────────────────
        sweep_uniform = self.uniform_sweeper.sweep(
            vit_preds_val, gt_val, raw_acc_val, eps=0.05)
        self.uniform_sweeper.plot(sweep_uniform, test_subject, self.figures_dir)

        best_perstate = self.perstate_sweeper.grid_search(
            vit_preds_val, gt_val, raw_acc_val, eps=0.03)

        print(f"\n  Uniform best d_min : {sweep_uniform['best_dmin']} frames "
              f"({sweep_uniform['best_dmin'] / self.sr * 1000:.0f} ms)")
        print(f"  Per-state best     : {best_perstate}")

        # ── Build all filters ─────────────────────────────────
        filt_dummy     = DurationFilter({c: 1 for c in range(self.num_classes)}, self.sr)
        filt_est       = DurationFilter(d_min_est, self.sr)
        filt_hardcoded = DurationFilter(self.dmin_hardcoded, self.sr)
        filt_uniform   = DurationFilter(
            {c: sweep_uniform['best_dmin'] for c in range(self.num_classes)}, self.sr)
        filt_perstate  = DurationFilter(best_perstate, self.sr)

        # ── Apply filters to TEST predictions only ────────────
        preds_vit_dur       = filt_est.apply(vit_preds)
        preds_vit_hardcoded = filt_hardcoded.apply(vit_preds)
        preds_vit_uniform   = filt_uniform.apply(vit_preds)
        preds_vit_perstate  = filt_perstate.apply(vit_preds)

        # ── Evaluate all methods on test subject ──────────────
        print(f"\n  Results on test subject {test_subject}:")
        m_raw           = self._compute_metrics(raw_preds,            gt_test, filt_dummy,     'Raw argmax')
        m_viterbi       = self._compute_metrics(vit_preds,            gt_test, filt_dummy,     'Viterbi only')
        m_vit_dur       = self._compute_metrics(preds_vit_dur,        gt_test, filt_est,       'Vit + d_min Q10 est.')
        m_vit_hardcoded = self._compute_metrics(preds_vit_hardcoded,  gt_test, filt_hardcoded, 'Vit + d_min Q10 hardcoded')
        m_vit_uniform   = self._compute_metrics(preds_vit_uniform,    gt_test, filt_uniform,   'Vit + d_min uniform sweep')
        m_vit_perstate  = self._compute_metrics(preds_vit_perstate,   gt_test, filt_perstate,  'Vit + d_min per-state')

        return dict(
            test_subject    = test_subject,
            d_min_estimated = d_min_est,
            d_min_perstate  = best_perstate,
            sweep_uniform   = sweep_uniform,
            raw             = m_raw,
            viterbi         = m_viterbi,
            vit_dur         = m_vit_dur,
            vit_hardcoded   = m_vit_hardcoded,
            vit_uniform     = m_vit_uniform,
            vit_perstate    = m_vit_perstate,
        )

    # ── Full LOSO run ─────────────────────────────────────────

    def run(self) -> List[dict]:
        all_subjects = sorted(f.stem for f in self.data_dir.glob('*.csv'))
        print(f"Subjects : {all_subjects}")
        print(f"Model    : {self.model_type.upper()}")

        all_results = []
        for subj in all_subjects:
            all_results.append(self._evaluate_fold(subj, all_subjects))

        self._print_final_summary(all_results)
        self._save_txt_report(all_results)
        self._plot_comparison(all_results)

        json_path = self.results_dir / \
            f'postfilter_{self.model_type}_{self.timestamp}.json'
        with open(json_path, 'w') as fh:
            json.dump(all_results, fh, indent=2, default=str)
        print(f"\n✓ JSON → {json_path}")
        return all_results

    # ── Summary helpers ───────────────────────────────────────

    @staticmethod
    def _mean_metric(all_results: List[dict], method: str, key: str) -> float:
        vals = [r[method][key] for r in all_results
                if isinstance(r.get(method), dict) and
                not np.isnan(r[method].get(key, float('nan')))]
        return float(np.mean(vals)) if vals else float('nan')

    def _print_final_summary(self, all_results: List[dict]) -> None:
        mm = self._mean_metric

        print(f"\n{'='*92}")
        print(f"  FINAL SUMMARY — {self.model_type.upper()}  "
              f"({len(all_results)} subjects)")
        print(f"{'='*92}")
        print(f"  {'Method':<44} {'Accuracy':>10} {'F1':>8} "
              f"{'FC/min':>8} {'|Lat| ms':>10} {'Det%':>7}")
        print("  " + "-"*90)
        for m in self.METHODS:
            acc = mm(all_results, m, 'accuracy')
            f1  = mm(all_results, m, 'f1_macro')
            fc  = mm(all_results, m, 'fc_per_min')
            lat = mm(all_results, m, 'median_abs_ms')
            det = mm(all_results, m, 'detection_rate') * 100
            print(f"  {self.LABELS[m]:<44} {acc:>10.4f} {f1:>8.4f} "
                  f"{fc:>8.2f} {lat:>10.1f} {det:>6.1f}%")

        print("\n  Improvements vs Raw argmax:")
        print("  " + "-"*90)
        acc_raw = mm(all_results, 'raw', 'accuracy')
        fc_raw  = mm(all_results, 'raw', 'fc_per_min')
        for m in ('vit_hardcoded', 'vit_uniform', 'vit_perstate'):
            acc_m = mm(all_results, m, 'accuracy')
            fc_m  = mm(all_results, m, 'fc_per_min')
            lat_m = mm(all_results, m, 'median_abs_ms')
            print(f"  {self.LABELS[m]}")
            print(f"    Δ Accuracy   : {acc_m - acc_raw:+.4f} "
                  f"({(acc_m - acc_raw) / acc_raw * 100:+.2f}%)")
            if fc_raw > 0:
                print(f"    FC reduction : {(fc_raw - fc_m) / fc_raw * 100:.1f}%")
            print(f"    |Lat| median : {lat_m:.1f} ms")

    def _plot_comparison(self, all_results: List[dict]) -> None:
        methods = ['raw', 'vit_hardcoded', 'vit_uniform', 'vit_perstate']
        labels  = [self.LABELS[m] for m in methods]
        mm      = self._mean_metric

        accs = [mm(all_results, m, 'accuracy')      for m in methods]
        fcs  = [mm(all_results, m, 'fc_per_min')    for m in methods]
        lats = [mm(all_results, m, 'median_abs_ms') for m in methods]

        colours = ['#8d8d8d', '#e07070', '#5b9bd5', '#70b870']
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        x = np.arange(len(methods))

        for ax, vals, title, ylabel in zip(
            axes,
            [accs, fcs, lats],
            ['Accuracy', 'FC / min', 'Median |Latency| (ms)'],
            ['Accuracy', 'False changes / min', 'ms']
        ):
            bars = ax.bar(x, vals, color=colours, alpha=0.85, edgecolor='white')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.01,
                        f'{v:.3f}' if title == 'Accuracy' else f'{v:.2f}',
                        ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        out = self.figures_dir / 'method_comparison.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Comparison plot → {out}")

    def _save_txt_report(self, all_results: List[dict]) -> None:
        mm  = self._mean_metric
        sep = "=" * 92
        txt_path = self.results_dir / \
            f'postfilter_report_{self.model_type}_{self.timestamp}.txt'

        with open(txt_path, 'w', encoding='utf-8') as f:
            def w(line=''):
                f.write(line + '\n')

            w(sep)
            w(f"POST-FILTER EVALUATION REPORT — {self.model_type.upper()}")
            w(sep)
            w(f"Generated : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
            w(f"Model     : {self.model_type.upper()}")
            w(f"Subjects  : {len(all_results)}")
            w()

            # Hardcoded d_min values
            w(sep); w("HARDCODED d_min (Q10 from EDA)"); w(sep)
            for state, sec in self.cfg['postfilter']['hardcoded_dmin_sec'].items():
                w(f"  {state}: {sec:.2f}s ({int(sec * self.sr)} frames)")
            w()

            # Summary table
            w(sep); w("SUMMARY — Mean over LOSO folds"); w(sep)
            w(f"  {'Method':<44} {'Accuracy':>10} {'F1':>10} "
              f"{'FC/min':>8} {'|Lat|ms':>10} {'Det%':>7}")
            w("  " + "-"*90)
            for m in self.METHODS:
                acc = mm(all_results, m, 'accuracy')
                f1  = mm(all_results, m, 'f1_macro')
                fc  = mm(all_results, m, 'fc_per_min')
                lat = mm(all_results, m, 'median_abs_ms')
                det = mm(all_results, m, 'detection_rate') * 100
                w(f"  {self.LABELS[m]:<44} {acc:>10.4f} {f1:>10.4f} "
                  f"{fc:>8.2f} {lat:>10.1f} {det:>6.1f}%")
            w()

            # Improvements vs raw
            w(sep); w("IMPROVEMENTS vs Raw argmax"); w(sep)
            acc_raw = mm(all_results, 'raw', 'accuracy')
            fc_raw  = mm(all_results, 'raw', 'fc_per_min')
            for m in ('vit_hardcoded', 'vit_uniform', 'vit_perstate'):
                acc_m = mm(all_results, m, 'accuracy')
                fc_m  = mm(all_results, m, 'fc_per_min')
                lat_m = mm(all_results, m, 'median_abs_ms')
                w(f"  {self.LABELS[m]}:")
                w(f"    Δ Accuracy   : {acc_m - acc_raw:+.4f} "
                  f"({(acc_m - acc_raw) / acc_raw * 100:+.2f}%)")
                if fc_raw > 0:
                    w(f"    FC reduction : {(fc_raw - fc_m) / fc_raw * 100:.1f}%")
                w(f"    |Lat| median : {lat_m:.1f} ms")
                w()

            # Per-subject breakdown
            w(sep); w("PER-SUBJECT BREAKDOWN"); w(sep)
            for r in all_results:
                sid = r['test_subject']
                w(f"\n  Subject: {sid}")
                w(f"    Uniform d_min   : {r['sweep_uniform']['best_dmin']} frames")
                w(f"    Per-state d_min : {r['d_min_perstate']}")
                w(f"\n  {'Method':<44} {'Accuracy':>10} {'F1':>8} "
                  f"{'FC/min':>8} {'|Lat|ms':>10}")
                w("  " + "-"*82)
                for m in self.METHODS:
                    rd  = r[m]
                    lat = (f"{rd['median_abs_ms']:.1f}"
                           if not np.isnan(rd['median_abs_ms']) else "N/A")
                    w(f"  {self.LABELS[m]:<44} {rd['accuracy']:>10.4f} "
                      f"{rd['f1_macro']:>8.4f} {rd['fc_per_min']:>8.2f} "
                      f"{lat:>10}")

            w(); w(sep); w("END OF REPORT"); w(sep)

        print(f"✓ Report → {txt_path}")