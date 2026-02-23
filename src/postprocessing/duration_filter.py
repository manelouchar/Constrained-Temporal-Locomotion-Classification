import numpy as np
from typing import Dict, List, Tuple

from src.postprocessing.transition_mask import STATES

class DurationFilter:
    def __init__(self, d_min: Dict[int, int], sampling_rate: int = 100):
        self.d_min = d_min
        self.sr    = sampling_rate

    def apply(self, predictions: np.ndarray) -> np.ndarray:
        """(T,) → (T,)"""
        T          = len(predictions)
        confirmed  = np.empty(T, dtype=int)
        cur_state  = int(predictions[0])
        cand_state = int(predictions[0])
        cand_count = 1
        confirmed[0] = cur_state

        for t in range(1, T):
            pred = int(predictions[t])
            if pred == cand_state:
                cand_count += 1
            else:
                cand_state = pred
                cand_count = 1
            if cand_count >= self.d_min.get(cand_state, 1) \
                    and cand_state != cur_state:
                cur_state = cand_state
            confirmed[t] = cur_state
        return confirmed

    def apply_batch(self, predictions: np.ndarray) -> np.ndarray:
        """(B, T) → (B, T)"""
        return np.stack([self.apply(predictions[b]) for b in range(len(predictions))])

    @staticmethod
    def estimate_from_labels(labels: np.ndarray, num_classes: int,
                             quantile: float = 0.25,
                             sampling_rate: int = 100) -> Dict[int, int]:
        runs: List[Tuple[int, int]] = []
        cur, cnt = int(labels[0]), 1
        for v in labels[1:]:
            if int(v) == cur:
                cnt += 1
            else:
                runs.append((cur, cnt))
                cur, cnt = int(v), 1
        runs.append((cur, cnt))

        lengths: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
        for state, length in runs:
            if state < num_classes:
                lengths[state].append(length)

        d_min = {}
        print(f"\n  d_min estimation (Q{quantile*100:.0f} of run lengths):")
        print(f"  {'State':>6}  {'N runs':>7}  {'Median':>8}  "
              f"{'d_min (fr)':>11}  {'d_min (ms)':>11}")
        print("  " + "-" * 52)
        for c in range(num_classes):
            lens = lengths[c]
            if not lens:
                d_min[c] = 1
                continue
            med      = int(np.median(lens))
            qval     = max(int(np.quantile(lens, quantile)), 1)
            d_min[c] = qval
            print(f"  {STATES[c]:>6}  {len(lens):>7}  {med:>8}  "
                  f"{qval:>11}  {qval/sampling_rate*1000:>10.0f}ms")
        return d_min

    def false_changes_per_min(self, predictions: np.ndarray,
                              ground_truth: np.ndarray) -> float:
        T             = len(predictions)
        total_minutes = T / self.sr / 60.0
        tolerance     = self.sr

        pred_trans = {t for t in range(1, T)
                      if predictions[t] != predictions[t-1]}
        true_trans = {t for t in range(1, T)
                      if ground_truth[t] != ground_truth[t-1]}

        false = sum(1 for pt in pred_trans
                    if not any(abs(pt - tt) <= tolerance for tt in true_trans))
        return false / total_minutes if total_minutes > 0 else 0.0

    def transition_latency(self, predictions: np.ndarray,
                           ground_truth: np.ndarray) -> Dict[str, float]:
        T      = len(ground_truth)
        window = self.sr * 3

        true_trans = [(t, int(ground_truth[t-1]), int(ground_truth[t]))
                      for t in range(1, T)
                      if ground_truth[t] != ground_truth[t-1]]

        latencies = []
        for t_true, _, s_to in true_trans:
            for t_pred in range(t_true, min(t_true + window, T)):
                if (t_pred > 0
                        and predictions[t_pred] != predictions[t_pred-1]
                        and predictions[t_pred] == s_to):
                    latencies.append(t_pred - t_true)
                    break

        if not latencies:
            return {'median_ms': np.nan, 'mean_ms': np.nan,
                    'iqr_ms': np.nan, 'n': 0}

        lat_ms   = np.array(latencies) / self.sr * 1000
        q25, q75 = np.percentile(lat_ms, [25, 75])
        return {'median_ms': float(np.median(lat_ms)),
                'mean_ms':   float(np.mean(lat_ms)),
                'iqr_ms':    float(q75 - q25),
                'n':         len(latencies)}
