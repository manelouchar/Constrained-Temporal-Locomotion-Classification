import numpy as np
from typing import Tuple, List

class WindowGenerator:
    """Generate sliding windows for LSTM training"""
    
    def __init__(self, window_size=2.5, overlap=0.5, sampling_rate=100):
        self.window_size_samples = int(window_size * sampling_rate)
        self.step_size_samples = int(self.window_size_samples * (1 - overlap))
        self.sampling_rate = sampling_rate
        
    def create_windows_sequence_labeling(self, imu_data, labels):
        windows, window_labels = [], []
        for start in range(0, len(imu_data) - self.window_size_samples + 1, 
                          self.step_size_samples):
            windows.append(imu_data[start:start + self.window_size_samples])
            window_labels.append(labels[start:start + self.window_size_samples])
        return np.array(windows), np.array(window_labels)


