import numpy as np
from src.postprocessing.transition_mask import LOG_TRANS

class ViterbiDecoder:
    def decode(self, logits: np.ndarray) -> np.ndarray:
        """(T, C) → (T,)"""
        T, C  = logits.shape
        lp    = self._log_softmax(logits)
        dp    = np.full((T, C), -np.inf)
        bp    = np.zeros((T, C), dtype=int)
        dp[0] = lp[0]

        for t in range(1, T):
            scores = dp[t-1, :, None] + LOG_TRANS   # (C, C)
            bp[t]  = scores.argmax(axis=0)
            dp[t]  = scores.max(axis=0) + lp[t]

        path      = np.empty(T, dtype=int)
        path[T-1] = dp[T-1].argmax()
        for t in range(T-2, -1, -1):
            path[t] = bp[t+1, path[t+1]]
        return path

    def decode_batch(self, logits: np.ndarray) -> np.ndarray:
        """(B, T, C) → (B, T)"""
        return np.stack([self.decode(logits[b]) for b in range(len(logits))])

    @staticmethod
    def argmax_baseline(logits: np.ndarray) -> np.ndarray:
        return np.argmax(logits, axis=-1)

    @staticmethod
    def _log_softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=-1, keepdims=True)
        return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))

