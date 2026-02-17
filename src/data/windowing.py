import numpy as np
from typing import Tuple, List

class WindowGenerator:
    """Génère des fenêtres glissantes pour l'entraînement LSTM"""
    
    def __init__(self, window_size: float = 2.0, overlap: float = 0.5, 
                 sampling_rate: int = 100):
        """
        Args:
            window_size: taille de la fenêtre en secondes
            overlap: pourcentage de chevauchement (0.0-1.0)
            sampling_rate: taux d'échantillonnage (Hz)
        """
        self.window_size_samples = int(window_size * sampling_rate)
        self.step_size_samples = int(self.window_size_samples * (1 - overlap))
        self.sampling_rate = sampling_rate
        
        print(f"Window config: {self.window_size_samples} samples, "
              f"step: {self.step_size_samples} samples")
    
    def create_windows(self, imu_data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée des fenêtres glissantes
        
        Args:
            imu_data: (N, n_features)
            labels: (N,)
            
        Returns:
            X_windows: (n_windows, window_size, n_features)
            y_windows: (n_windows,) - label majoritaire dans chaque fenêtre
        """
        n_samples = len(imu_data)
        n_features = imu_data.shape[1]
        
        windows = []
        window_labels = []
        
        for start in range(0, n_samples - self.window_size_samples + 1, self.step_size_samples):
            end = start + self.window_size_samples
            
            # Extraire fenêtre
            window = imu_data[start:end]
            window_label_seq = labels[start:end]
            
            # Label = mode majoritaire dans la fenêtre
            # (stratégie simple, peut être améliorée)
            unique, counts = np.unique(window_label_seq, return_counts=True)
            majority_label = unique[np.argmax(counts)]
            
            windows.append(window)
            window_labels.append(majority_label)
        
        X_windows = np.array(windows)
        y_windows = np.array(window_labels)
        
        return X_windows, y_windows
    
    def create_windows_sequence_labeling(self, imu_data: np.ndarray, 
                                         labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée des fenêtres avec labels séquentiels (pour LSTM many-to-many)
        
        Returns:
            X_windows: (n_windows, window_size, n_features)
            y_windows: (n_windows, window_size) - label à chaque timestep
        """
        n_samples = len(imu_data)
        
        windows = []
        window_labels = []
        
        for start in range(0, n_samples - self.window_size_samples + 1, self.step_size_samples):
            end = start + self.window_size_samples
            
            window = imu_data[start:end]
            window_label_seq = labels[start:end]
            
            windows.append(window)
            window_labels.append(window_label_seq)
        
        return np.array(windows), np.array(window_labels)