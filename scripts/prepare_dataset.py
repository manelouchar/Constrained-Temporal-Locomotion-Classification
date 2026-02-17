import pandas as pd
from pathlib import Path
import yaml

# Load config
CONFIG_PATH = Path("config/config.yaml")
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DATA_DIR = Path(CONFIG['paths']['data_dir'])
PROCESSED_DIR = Path(CONFIG['paths']['processed_dir'])
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

N_SUBJECTS = CONFIG['data']['n_subjects']

# Mapping transitions -> target stable mode
TRANSITION_TO_TARGET = {
    12: 20,  # LW→RA -> RA
    13: 30,  # LW→RD -> RD
    14: 40,  # LW→SA -> SA
    15: 50,  # LW→SD -> SD
    21: 10,  # RA→LW -> LW
    31: 10,  # RD→LW -> LW
    41: 10,  # SA→LW -> LW
    51: 10,  # SD→LW -> LW
}

# Stable classes -> 0-4
STABLE_TO_INDEX = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4}

# Process each subject
SUBJECTS = [f"S{i+1}" for i in range(N_SUBJECTS)]

for subj in SUBJECTS:
    file_path = DATA_DIR / f"{subj}.csv"
    df = pd.read_csv(file_path)
    
    # Map transition labels → target mode
    df['label'] = df['label'].replace(TRANSITION_TO_TARGET)
    
    # Keep only stable classes
    df = df[df['label'].isin(STABLE_TO_INDEX.keys())].reset_index(drop=True)
    
    # Map stable labels → 0-4 
    df['label_idx'] = df['label'].map(STABLE_TO_INDEX)
    
    # Save the entire dataframe as-is
    df.to_csv(PROCESSED_DIR / f"{subj}.csv", index=False)
    
    print(f"{subj}: {len(df)} frames saved in {PROCESSED_DIR / f'{subj}.csv'}")

print("Dataset preprocessing completed. All subjects saved in", PROCESSED_DIR)