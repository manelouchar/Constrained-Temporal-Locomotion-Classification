import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt


class IMUPreprocessor:
    """
    IMU preprocessing:
    - Optional low-pass filtering (Butterworth)
    - Z-score normalization per subject (fit on TRAIN only)
    """

    def __init__(
        self,
        method: str = "zscore",
        use_lowpass: bool = True,
        cutoff_hz: float = 25.0,
        sampling_rate: int = 100,
        filter_order: int = 4
    ):
        self.method = method
        self.use_lowpass = use_lowpass
        self.cutoff_hz = cutoff_hz
        self.sampling_rate = sampling_rate
        self.filter_order = filter_order

        self.scalers = {}  # one scaler per subject

    # ---------------------------------------------------
    # Low-pass filter
    # ---------------------------------------------------
    def _apply_lowpass(self, X: np.ndarray) -> np.ndarray:
        if not self.use_lowpass:
            return X

        nyquist = self.sampling_rate / 2.0
        normal_cutoff = self.cutoff_hz / nyquist
        b, a = butter(self.filter_order, normal_cutoff, btype="low")

        X_filt = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_filt[:, i] = filtfilt(b, a, X[:, i])

        return X_filt

    # ---------------------------------------------------
    # Fit scaler on TRAIN data only
    # ---------------------------------------------------
    def fit(self, X: np.ndarray, subject_id: str):
        """
        Fit normalization parameters for one subject (TRAIN only)
        """
        X = self._apply_lowpass(X)

        if self.method == "zscore":
            scaler = StandardScaler()
            scaler.fit(X)
            self.scalers[subject_id] = scaler
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    # ---------------------------------------------------
    # Transform using fitted scaler
    # ---------------------------------------------------
    def transform(self, X: np.ndarray, subject_id: str) -> np.ndarray:
        """
        Apply filtering + normalization
        """
        if subject_id not in self.scalers:
            raise ValueError(f"Scaler not fitted for subject {subject_id}")

        X = self._apply_lowpass(X)
        return self.scalers[subject_id].transform(X)

    # ---------------------------------------------------
    # Fit + transform shortcut
    # ---------------------------------------------------
    def fit_transform(self, X: np.ndarray, subject_id: str) -> np.ndarray:
        self.fit(X, subject_id)
        return self.transform(X, subject_id)
