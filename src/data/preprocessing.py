import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

class IMUPreprocessor:
    """IMU preprocessing: optional low-pass filtering + Z-score normalization"""
    
    def __init__(self, method='zscore', use_lowpass=False,
                 cutoff_hz=25.0, sampling_rate=100, filter_order=4):
        from scipy.signal import butter, filtfilt
        
        self.method        = method
        self.use_lowpass   = use_lowpass
        self.cutoff_hz     = cutoff_hz
        self.sampling_rate = sampling_rate
        self.filter_order  = filter_order
        self.scalers       = {}
        self.butter        = butter
        self.filtfilt      = filtfilt

    def _apply_lowpass(self, X):
        if not self.use_lowpass:
            return X
        from sklearn.preprocessing import StandardScaler
        nyquist = self.sampling_rate / 2.0
        normal_cutoff = self.cutoff_hz / nyquist
        b, a = self.butter(self.filter_order, normal_cutoff, btype="low")
        X_filt = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_filt[:, i] = self.filtfilt(b, a, X[:, i])
        return X_filt

    def fit(self, X, subject_id):
        from sklearn.preprocessing import StandardScaler
        X = self._apply_lowpass(X)
        if self.method == "zscore":
            scaler = StandardScaler()
            scaler.fit(X)
            self.scalers[subject_id] = scaler
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def transform(self, X, subject_id):
        if subject_id not in self.scalers:
            raise ValueError(f"Scaler not fitted for subject {subject_id}")
        X = self._apply_lowpass(X)
        return self.scalers[subject_id].transform(X)

    def fit_transform(self, X, subject_id):
        self.fit(X, subject_id)
        return self.transform(X, subject_id)
