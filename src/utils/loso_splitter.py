import numpy as np
import pandas as pd
import json
import yaml
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut

class LOSOSplitter:
    def __init__(self, config_path='../../config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(self.config['paths']['data_dir'])
        self.loso_dir = Path(self.config['paths']['loso_dir'])
        self.loso_dir.mkdir(parents=True, exist_ok=True)
    
    def load_all_subjects(self):
        """Load all subject data"""
        subjects = {}
        subject_files = list(self.data_dir.glob('S*.csv'))
        
        for file_path in subject_files:
            subject_id = file_path.stem  # S1, S2, etc.
            df = pd.read_csv(file_path)
            df['subject'] = subject_id
            subjects[subject_id] = df
        
        print(f"Loaded {len(subjects)} subjects")
        return subjects
    
    def create_loso_splits(self, save=True):
        """Create Leave-One-Subject-Out splits"""
        subjects = self.load_all_subjects()
        subject_ids = sorted(subjects.keys())
        
        splits = {}
        
        for i, test_subject in enumerate(subject_ids):
            fold_name = f"fold_{i}"
            
            # Train subjects: all except test subject
            train_subjects = [s for s in subject_ids if s != test_subject]
            
            # Create split dictionary
            splits[fold_name] = {
                'train_subjects': train_subjects,
                'test_subject': test_subject,
                'fold_id': i,
                'num_train_subjects': len(train_subjects),
                'num_test_subject': 1
            }
            
            print(f"Fold {i}: Test={test_subject}, Train={train_subjects}")
            
            # Optionally save data for this fold
            if save:
                self.save_fold_data(fold_name, subjects, train_subjects, test_subject)
        
        # Save splits metadata
        if save:
            with open(self.loso_dir / 'loso_splits_metadata.json', 'w') as f:
                json.dump(splits, f, indent=2)
            
            print(f"\nSaved LOSO splits to {self.loso_dir}")
        
        return splits
    
    def save_fold_data(self, fold_name, subjects, train_subjects, test_subject):
        """Save preprocessed data for a specific fold"""
        fold_dir = self.loso_dir / fold_name
        fold_dir.mkdir(exist_ok=True)
        
        # Save train subjects
        train_dfs = []
        for subject in train_subjects:
            train_dfs.append(subjects[subject])
        
        if train_dfs:
            train_df = pd.concat(train_dfs, ignore_index=True)
            train_df.to_csv(fold_dir / 'train.csv', index=False)
        
        # Save test subject
        if test_subject in subjects:
            test_df = subjects[test_subject]
            test_df.to_csv(fold_dir / 'test.csv', index=False)
        
        # Save metadata
        metadata = {
            'fold_name': fold_name,
            'train_subjects': train_subjects,
            'test_subject': test_subject,
            'num_train_samples': len(train_df) if 'train_df' in locals() else 0,
            'num_test_samples': len(test_df) if 'test_df' in locals() else 0
        }
        
        with open(fold_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_fold(self, fold_id):
        """Load a specific fold"""
        fold_name = f"fold_{fold_id}"
        fold_dir = self.loso_dir / fold_name
        
        if not fold_dir.exists():
            raise ValueError(f"Fold {fold_id} not found. Run create_loso_splits first.")
        
        train_df = pd.read_csv(fold_dir / 'train.csv')
        test_df = pd.read_csv(fold_dir / 'test.csv')
        
        with open(fold_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return train_df, test_df, metadata

if __name__ == "__main__":
    splitter = LOSOSplitter()
    splits = splitter.create_loso_splits(save=True)
    print(f"\nCreated {len(splits)} LOSO folds")