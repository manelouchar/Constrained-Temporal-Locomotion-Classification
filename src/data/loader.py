import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

class IMUDataLoader:
    """Load dataset"""
    
    def __init__(self, config_path: str = '../config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.classes = self.config['data']['classes']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Labels mapping
        self.label_mapping = {
            # Gait modes
            10: 0,  
            20: 3,    
            30: 4,  
            40: 1,  
            50: 2,  
            
            # Transitions LW → X (1X)
            12: 3,  # LW→RA (transition) → RA
            13: 4,  # LW→RD (transition) → RD
            14: 1,  # LW→SA (transition) → SA
            15: 2,  # LW→SD (transition) → SD
            
            # Transitions X → LW (X1)
            21: 0,  # RA→LW (transition) → LW
            31: 0,  # RD→LW (transition) → LW
            41: 0,  # SA→LW (transition) → LW
            51: 0   # SD→LW (transition) → LW
        }
        
        # IMU columns
        self.imu_columns = [
            # Accelerometers (6)
            'x_acc_left', 'y_acc_left', 'z_acc_left',
            'x_acc_right', 'y_acc_right', 'z_acc_right',
            # Quaternions left (4)
            'quat_1_left', 'quat_2_left', 'quat_3_left', 'quat_4_left',
            # Quaternions right (4)
            'quat_1_right', 'quat_2_right', 'quat_3_right', 'quat_4_right',
            # Gyroscopes (6)
            'x_gyro_left', 'y_gyro_left', 'z_gyro_left',
            'x_gyro_right', 'y_gyro_right', 'z_gyro_right',
            # Kalman features (7)
            'feature1', 'feature2', 'feature3', 'feature4',
            'feature5', 'feature6', 'feature7'
        ]
    
    def load_subject_data(self, subject_id: int, 
                          data_dir: str = 'data/combined') -> Dict:
        """
        Loads subject data from SX.csv
        """
        filepath = Path(data_dir) / f'S{subject_id}.csv'
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
    
        # load csv
        df = pd.read_csv(filepath)
        
        # Check that columns exist  
        missing_cols = [col for col in self.imu_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")
        
        # Extract IMU data (27 features)
        imu_data = df[self.imu_columns].values.astype(np.float32)
        
        # Extract raw labels
        raw_labels = df['label'].values.astype(np.int32)
        
        # Convert to simple labels 
        simplified_labels = np.array([
            self.label_mapping.get(label, -1) for label in raw_labels
        ])
        
        # Check that there are no unknown labels
        if np.any(simplified_labels == -1):
            unknown_labels = np.unique(raw_labels[simplified_labels == -1])
            raise ValueError(f"Unknown labels found: {unknown_labels}")
        
        # Extract timestamps
        if 'time' in df.columns:
            timestamps = df['time'].values.astype(np.float32)
        else:
            sampling_rate = self.config['data']['sampling_rate']
            timestamps = np.arange(len(imu_data), dtype=np.float32) / sampling_rate
        
        return {
            'imu': imu_data,
            'labels': simplified_labels,
            'raw_labels': raw_labels,
            'timestamps': timestamps,
            'subject_id': subject_id
        }
    
    def load_all_subjects(self, data_dir: str = 'data/combined') -> List[Dict]:
        """Load all subjects (S1-S10)"""
        n_subjects = self.config['data']['n_subjects']
        all_data = []
        
        print("Loading subjects...")
        for subject_id in range(1, n_subjects + 1):
            try:
                data = self.load_subject_data(subject_id, data_dir)
                all_data.append(data)
                print(f"  ✓ Subject {subject_id}: {len(data['imu']):,} samples, "
                      f"duration: {data['timestamps'][-1]:.1f}s")
            except FileNotFoundError:
                print(f"  ✗ Subject {subject_id}: File not found")
            except Exception as e:
                print(f"  ✗ Subject {subject_id}: Error - {e}")
        
        print(f"\nTotal: {len(all_data)} subjects loaded")
        return all_data