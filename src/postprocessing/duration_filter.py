import numpy as np
from typing import Dict, List, Tuple

from src.postprocessing.transition_mask import STATES

SR          = 100 

class DurationFilter:
    """
    Minimum-duration confirmation filter.

    Accepts a new locomotion mode only after d_min[state] consecutive
    frames of agreement, encoding the physiological constraint that
    a genuine mode change requires a minimum biomechanical commitment
    time before it is mechanically established.

    Per-state d_min values allow stair/ramp transitions (which develop
    over longer biomechanical timescales) to require longer confirmation
    than level-walking transitions.
    """

    def __init__(self, d_min: Dict[int, int], sampling_rate: int = SR):
        self.d_min = d_min
        self.sr    = sampling_rate

    # ── Core filter ──────────────────────────────────────────

    def apply(self, predictions: np.ndarray) -> np.ndarray:
        """
        Args:
            predictions: (T,) integer label sequence (e.g. from Viterbi)
        Returns:
            confirmed: (T,) filtered label sequence
        """
        T            = len(predictions)
        confirmed    = np.empty(T, dtype=int)
        cur_state    = int(predictions[0])
        cand_state   = int(predictions[0])
        cand_count   = 1
        confirmed[0] = cur_state

        for t in range(1, T):
            pred = int(predictions[t])
            if pred == cand_state:
                cand_count += 1
            else:
                cand_state = pred
                cand_count = 1

            # Promote candidate to confirmed state once threshold met
            if (cand_state != cur_state and
                    cand_count >= self.d_min.get(cand_state, 1)):
                cur_state = cand_state
            confirmed[t] = cur_state
        return confirmed

    # ── d_min estimation from training labels ────────────────

    @staticmethod
    def estimate_from_labels(labels: np.ndarray,
                             subject_boundaries: List[int],
                             num_classes: int,
                             quantile: float = 0.10,
                             sampling_rate: int = SR) -> Dict[int, int]:
        """
        Estimate d_min per class from run-length distributions of the
        TRAINING labels.

        CORRECTION vs. original: subject boundary frames are stripped
        before computing run lengths to avoid artefactual short runs
        created where two subjects are concatenated.

        Args:
            labels:             (N,) concatenated training labels
            subject_boundaries: list of indices where subject changes occur
            num_classes:        K
            quantile:           Q10 = conservative lower bound, avoids
                                over-smoothing while suppressing short noise
            sampling_rate:      Hz

        Returns:
            d_min: {class_idx: min_frames}
        """
        boundary_set = set(subject_boundaries)

        # Run-length encoding, skipping boundary transitions
        runs: List[Tuple[int, int]] = []
        cur, cnt = int(labels[0]), 1
        for t in range(1, len(labels)):
            if t in boundary_set:
                # flush current run, do not create boundary run
                runs.append((cur, cnt))
                cur, cnt = int(labels[t]), 1
            elif int(labels[t]) == cur:
                cnt += 1
            else:
                runs.append((cur, cnt))
                cur, cnt = int(labels[t]), 1
        runs.append((cur, cnt))

        lengths: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
        for state, length in runs:
            if 0 <= state < num_classes:
                lengths[state].append(length)

        d_min: Dict[int, int] = {}
        print(f"\n  d_min estimation (Q{quantile*100:.0f} of run lengths):")
        print(f"  {'State':>6}  {'N runs':>7}  {'Median':>8}  "
              f"{'d_min(fr)':>10}  {'d_min(ms)':>10}")
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
                  f"{qval:>10}  {qval/sampling_rate*1000:>9.0f}ms")
        return d_min

    # ── FC/min ───────────────────────────────────────────────

    def false_changes_per_min(self,
                              predictions: np.ndarray,
                              ground_truth: np.ndarray,
                              tolerance_sec: float = 0.5) -> float:
        """
        False Changes per minute.

        Definition (publication standard):
            A predicted transition at time t is FALSE if there is no
            ground-truth transition within ±tolerance_sec of t.

        Normalisation: per minute of RECORDING TIME (not per transition),
        consistent with Moon et al. (2025) and Young et al. (2014).

        Args:
            predictions:    (T,) predicted sequence
            ground_truth:   (T,) ground-truth sequence
            tolerance_sec:  half-window in seconds for GT matching
                            (default 0.5 s = 50 frames at 100 Hz)
        Returns:
            fc_per_min: float ≥ 0
        """
        T             = len(predictions)
        duration_min  = T / self.sr / 60.0
        tolerance_fr  = int(tolerance_sec * self.sr)

        # Ground-truth transition times (frame indices)
        gt_trans = {t for t in range(1, T)
                    if ground_truth[t] != ground_truth[t-1]}

        false_count = 0
        for t in range(1, T):
            if predictions[t] != predictions[t-1]:
                # False iff no GT transition within ±tolerance_fr
                if not any(abs(t - tt) <= tolerance_fr for tt in gt_trans):
                    false_count += 1

        return false_count / duration_min if duration_min > 0 else 0.0

    # ── Transition latency ───────────────────────────────────

    def transition_latency(self,
                           predictions: np.ndarray,
                           ground_truth: np.ndarray,
                           max_window_sec: float = 3.0) -> Dict[str, float]:
        """
        Publication-grade transition detection latency.

        For each ground-truth transition (s_from → s_to) at frame t_true,
        find the closest predicted transition of the SAME type (same
        s_from and s_to pair) within ±max_window_sec.

        Latency (signed) = t_pred − t_true
            > 0 : delayed detection  (predicted AFTER ground truth)
            < 0 : early detection    (predicted BEFORE ground truth)

        The paper table reports:
            median_abs_ms  — median of |latency| in ms  (always ≥ 0)
            median_signed_ms — median of signed latency  (shows bias)

        One-to-one matching: each predicted transition is consumed once.

        Returns dict with keys:
            median_abs_ms, median_signed_ms,
            mean_abs_ms,   mean_signed_ms,
            iqr_abs_ms,    p25_abs_ms,  p75_abs_ms,
            n_detected,    n_missed,    detection_rate
        """
        T          = len(ground_truth)
        max_frames = int(self.sr * max_window_sec)

        # Extract ground-truth transitions
        true_trans: List[Tuple[int, int, int]] = []
        for t in range(1, T):
            if ground_truth[t] != ground_truth[t-1]:
                true_trans.append((t, int(ground_truth[t-1]), int(ground_truth[t])))

        # Extract predicted transitions
        pred_trans: List[Tuple[int, int, int]] = []
        for t in range(1, T):
            if predictions[t] != predictions[t-1]:
                pred_trans.append((t, int(predictions[t-1]), int(predictions[t])))

        used_pred: set = set()
        latencies_fr: List[int] = []
        missed = 0

        for t_true, s_from, s_to in true_trans:
            best_idx   = None
            best_dist  = None
            best_delay = None

            for i, (t_pred, p_from, p_to) in enumerate(pred_trans):
                if i in used_pred:
                    continue
                # Strict type match — prevents cross-type confusion
                if p_from != s_from or p_to != s_to:
                    continue
                delay = t_pred - t_true
                if abs(delay) <= max_frames:
                    if best_dist is None or abs(delay) < best_dist:
                        best_dist  = abs(delay)
                        best_delay = delay
                        best_idx   = i

            if best_idx is not None:
                latencies_fr.append(best_delay)
                used_pred.add(best_idx)
            else:
                missed += 1

        n_true = len(true_trans)
        n_det  = len(latencies_fr)

        if n_det == 0:
            nan = float('nan')
            return dict(median_abs_ms=nan, median_signed_ms=nan,
                        mean_abs_ms=nan,   mean_signed_ms=nan,
                        iqr_abs_ms=nan,    p25_abs_ms=nan, p75_abs_ms=nan,
                        n_detected=0, n_missed=missed,
                        detection_rate=0.0)

        lat_fr  = np.array(latencies_fr, dtype=float)
        lat_ms  = lat_fr / self.sr * 1000.0
        abs_ms  = np.abs(lat_ms)
        p25, p75 = np.percentile(abs_ms, [25, 75])

        return dict(
            median_abs_ms    = float(np.median(abs_ms)),
            median_signed_ms = float(np.median(lat_ms)),
            mean_abs_ms      = float(np.mean(abs_ms)),
            mean_signed_ms   = float(np.mean(lat_ms)),
            iqr_abs_ms       = float(p75 - p25),
            p25_abs_ms       = float(p25),
            p75_abs_ms       = float(p75),
            n_detected       = n_det,
            n_missed         = missed,
            detection_rate   = n_det / (n_true + 1e-9),
        )

    # ── Diagnostic breakdown ─────────────────────────────────

    def fc_breakdown(self,
                     predictions: np.ndarray,
                     ground_truth: np.ndarray,
                     tolerance_sec: float = 0.5) -> dict:
        """
        Diagnostic decomposition of predicted transitions into:
            matched   : TP transitions (near a GT event)
            false     : FP transitions (no nearby GT event)
        """
        T            = len(predictions)
        tolerance_fr = int(tolerance_sec * self.sr)
        duration_min = T / self.sr / 60.0

        gt_trans   = {t for t in range(1, T)
                      if ground_truth[t] != ground_truth[t-1]}
        pred_trans = [t for t in range(1, T)
                      if predictions[t] != predictions[t-1]]

        false_list, matched_list = [], []
        for t in pred_trans:
            if any(abs(t - tt) <= tolerance_fr for tt in gt_trans):
                matched_list.append(t)
            else:
                false_list.append(t)

        fc_per_min = len(false_list) / duration_min if duration_min > 0 else 0.0
        return dict(
            n_pred_trans   = len(pred_trans),
            n_gt_trans     = len(gt_trans),
            n_matched      = len(matched_list),
            n_false        = len(false_list),
            fc_per_min     = fc_per_min,
            duration_min   = duration_min,
        )
