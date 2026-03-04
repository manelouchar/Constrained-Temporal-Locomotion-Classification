import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.postprocessing.duration_filter import DurationFilter
from sklearn.metrics import accuracy_score

NUM_CLASSES = 5  # LW, SA, SD, RA, RD
SR          = 100 

class DMinSweeper:
    """
    Uniform d_min sweep: a single threshold shared across all states.

    Optimisation objective (constrained):
        minimise  2 × FC/min + Lat_median_abs_ms / 500
        subject to  Acc ≥ Acc_raw − ε   (ε = 0.02)

    The weight of 2 reflects the higher clinical cost of false actuations
    relative to detection latency in safety-critical control.
    Performed on the VALIDATION subject to avoid test leakage.
    """
    def __init__(self, dmin_range, sampling_rate: int = SR):
        self.dmin_range = list(dmin_range)
        self.sr         = sampling_rate

    def sweep(self,
              vit_preds_val:  np.ndarray,
              gt_val:         np.ndarray,
              raw_acc_val:    float,
              eps:            float = 0.02) -> dict:
        """
        All computations are on the VALIDATION split.

        Args:
            vit_preds_val : Viterbi predictions on val subject (T,)
            gt_val        : ground-truth labels on val subject (T,)
            raw_acc_val   : raw argmax accuracy on val subject
            eps           : max acceptable accuracy drop

        Returns:
            dict with sweep curves and best_dmin
        """
        acc_list, fc_list, lat_list = [], [], []

        for d in self.dmin_range:
            filt  = DurationFilter({c: d for c in range(NUM_CLASSES)}, self.sr)
            preds = filt.apply(vit_preds_val)

            acc = accuracy_score(gt_val, preds)
            fc  = filt.false_changes_per_min(preds, gt_val)
            lat = filt.transition_latency(preds, gt_val)
            lat_ms = lat['median_abs_ms'] if not np.isnan(lat['median_abs_ms']) else 0.0

            acc_list.append(acc)
            fc_list.append(fc)
            lat_list.append(lat_ms)

        # Accuracy-constrained candidates
        valid_idx = [i for i, a in enumerate(acc_list)
                     if a >= raw_acc_val - eps]
        if not valid_idx:
            valid_idx = [int(np.argmax(acc_list))]

        scores    = [fc_list[i] * 2.0 + lat_list[i] / 500.0 for i in valid_idx]
        best_i    = valid_idx[int(np.argmin(scores))]

        return dict(
            dmin_range   = self.dmin_range,
            accuracy     = acc_list,
            fc           = fc_list,
            lat          = lat_list,
            best_dmin    = int(self.dmin_range[best_i]),
            best_accuracy= acc_list[best_i],
            best_fc      = fc_list[best_i],
            best_lat     = lat_list[best_i],
        )

    def plot(self, sweep_result: dict, subject: str, out_dir: Path) -> None:
        dr  = sweep_result['dmin_range']
        acc = sweep_result['accuracy']
        fc  = sweep_result['fc']
        lat = sweep_result['lat']
        bd  = sweep_result['best_dmin']

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        ax1, ax2 = axes
        ax1r = ax1.twinx()
        ax1.plot(dr, fc,  'b-o', ms=4, lw=2, label='FC/min')
        ax1r.plot(dr, lat,'r-s', ms=4, lw=2, label='Lat |med| (ms)')
        ax1.axvline(x=bd, color='g', ls='--', lw=2, label=f'Best={bd} fr')
        ax1.set_xlabel('d_min (frames)', fontsize=11)
        ax1.set_ylabel('FC / min', color='b', fontsize=11)
        ax1r.set_ylabel('Median |Latency| (ms)', color='r', fontsize=11)
        ax1.set_title(f'd_min Sweep — FC & Latency — {subject}', fontweight='bold')
        ax1.grid(alpha=0.3)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1r.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper right')

        ax2.plot(dr, acc, 'g-^', ms=5, lw=2, label='Val Accuracy')
        ax2.axvline(x=bd, color='g', ls='--', lw=2, label=f'Best={bd} fr')
        ax2.set_xlabel('d_min (frames)', fontsize=11)
        ax2.set_ylabel('Accuracy', fontsize=11)
        ax2.set_title(f'd_min Sweep — Accuracy — {subject}', fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.legend(loc='lower left')

        plt.tight_layout()
        fig.savefig(out_dir / f'dmin_sweep_{subject}.png', dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

