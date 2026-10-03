"""
IMU Processing and Feature Extraction Script.
Processes all 1356 clinical gait IMU trials from Figshare dataset.
Extracts spatiotemporal and kinematic biomechanical features.
Saves:
  - data/features/real_imu_features.parquet
"""
import os
import sys
import glob
import time
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.imu.loader import create_standardized_imu_table, load_trial_metadata
from src.imu.preprocessor import preprocess_imu_dataframe
from src.imu.features import extract_trial_features
IMU_BASE = os.path.join(PROJECT_ROOT, 'data', 'real', 'imu', 'dataset', 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'features')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'imu')
METADATA_DIR = os.path.join(PROJECT_ROOT, 'metadata')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

splits_path = os.path.join(METADATA_DIR, 'imu_splits.csv')
df_splits = pd.read_csv(splits_path) if os.path.exists(splits_path) else None
split_map = dict(zip(df_splits['subject'], df_splits['split'])) if df_splits is not None else {}

def run_imu_feature_extraction():
    meta_files = glob.glob(os.path.join(IMU_BASE, '*', '*', '*', '*', '*_meta.json'))
    print(f"Found {len(meta_files)} IMU trials to process.")
    
    rows = []
    t0 = time.time()
    for idx, meta_p in enumerate(meta_files):
        trial_dir = os.path.dirname(meta_p)
        try:
            meta = load_trial_metadata(trial_dir)
            df_raw = create_standardized_imu_table(trial_dir)
            if len(df_raw) == 0:
                continue
            df_proc = preprocess_imu_dataframe(df_raw)
            feat = extract_trial_features(df_proc, meta)
            
            # Map split
            subj = feat['source_subject_id']
            feat['split'] = split_map.get(subj, 'TRAIN')
            
            # Save first 5 preprocessed sample files into data/processed/imu
            if idx < 5:
                sample_name = f"{os.path.basename(trial_dir)}_preprocessed.parquet"
                df_proc.to_parquet(os.path.join(PROCESSED_DIR, sample_name), index=False)
                
            rows.append(feat)
        except Exception as e:
            print(f"Error processing {trial_dir}: {e}")
            
        if (idx + 1) % 250 == 0 or (idx + 1) == len(meta_files):
            print(f"  Processed {idx + 1}/{len(meta_files)} trials in {time.time() - t0:.1f}s")
            
    df_all = pd.DataFrame(rows)
    out_parquet = os.path.join(OUTPUT_DIR, 'real_imu_features.parquet')
    df_all.to_parquet(out_parquet, index=False)
    print(f"\nSaved IMU features table ({df_all.shape[0]} rows, {df_all.shape[1]} cols) to {out_parquet}")
    print("\nClass breakdown:")
    print(df_all.groupby(['split', 'pathology_key'])['source_subject_id'].nunique().unstack(fill_value=0))
    return df_all

if __name__ == '__main__':
    run_imu_feature_extraction()
