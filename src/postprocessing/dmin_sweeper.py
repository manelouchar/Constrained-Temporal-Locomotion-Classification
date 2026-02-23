import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.postprocessing.duration_filter import DurationFilter
NUM_CLASSES = 5  # LW, SA, SD, RA, RD

class DMinSweeper:
    def __init__(self, dmin_range, sampling_rate: int = 100):
        self.dmin_range = list(dmin_range)
        self.sr         = sampling_rate

    def sweep(self, viterbi_preds: np.ndarray,
              ground_truth: np.ndarray) -> dict:
        fc_list, lat_list = [], []
        for d in self.dmin_range:
            filt  = DurationFilter({c: d for c in range(NUM_CLASSES)}, self.sr)
            preds = filt.apply(viterbi_preds)
            fc    = filt.false_changes_per_min(preds, ground_truth)
            lat   = filt.transition_latency(preds, ground_truth)
            fc_list.append(fc)
            lat_list.append(lat['median_ms'] if not np.isnan(lat['median_ms']) else 0.0)

        scores   = [fc + lat / 400.0 for fc, lat in zip(fc_list, lat_list)]
        best_idx = int(np.argmin(scores))
        return {'dmin_range': self.dmin_range, 'fc': fc_list,
                'lat': lat_list, 'best_dmin': int(self.dmin_range[best_idx])}

    def plot(self, sweep_result: dict, subject: str, out_dir: Path) -> None:
        dr, fc, lat = (sweep_result['dmin_range'],
                       sweep_result['fc'], sweep_result['lat'])
        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax2 = ax1.twinx()
        ax1.plot(dr, fc,  'b-o', markersize=4, label='FC/min')
        ax2.plot(dr, lat, 'r-s', markersize=4, label='Latency (ms)')
        ax1.axvline(x=sweep_result['best_dmin'], color='g', linestyle='--',
                    label=f"best={sweep_result['best_dmin']}fr")
        ax1.set_xlabel('d_min (frames)')
        ax1.set_ylabel('FC / min', color='b')
        ax2.set_ylabel('Median Latency (ms)', color='r')
        ax1.set_title(f'd_min sweep — {subject}', fontweight='bold')
        lines = (ax1.get_legend_handles_labels()[0] +
                 ax2.get_legend_handles_labels()[0])
        lbls  = (ax1.get_legend_handles_labels()[1] +
                 ax2.get_legend_handles_labels()[1])
        ax1.legend(lines, lbls, loc='upper right')
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f'dmin_sweep_{subject}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

