"""
Patient-Level Train / Validation / Test Splitting Script.
Ensures strict participant-level isolation with random_seed = 42.
Never leaks windows or trials across splits.
Outputs:
  - metadata/imu_splits.csv
  - metadata/semg_splits.csv
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
METADATA_DIR = os.path.join(PROJECT_ROOT, 'metadata')
os.makedirs(METADATA_DIR, exist_ok=True)

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==========================================
# 1. IMU PATIENT-LEVEL SPLIT
# ==========================================
imu_trial_idx_path = os.path.join(METADATA_DIR, 'imu_trial_index.csv')
if not os.path.exists(imu_trial_idx_path):
    print("Generating imu_trial_index.csv first...")
    import subprocess
    subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, 'src', 'index_imu_meta.py')], check=True)

df_imu = pd.read_csv(imu_trial_idx_path)
# Get unique subjects with their primary pathologyKey and group
imu_subjects = df_imu.groupby('subject').agg(
    group=('group', 'first'),
    pathologyKey=('pathologyKey', 'first'),
    pathology=('pathology', 'first'),
    trial_count=('trial', 'count')
).reset_index()

# Stratification label
strat_label = imu_subjects['pathologyKey'].copy()
# If any category has fewer than 3 subjects, map to cohort group for stratification
counts = strat_label.value_counts()
small_cats = counts[counts < 3].index
strat_label = strat_label.apply(lambda x: imu_subjects.loc[strat_label == x, 'group'].iloc[0] if x in small_cats else x)

# Split 70% train, 30% temp
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
train_idx, temp_idx = next(sss1.split(imu_subjects, strat_label))

train_subjs = imu_subjects.iloc[train_idx].copy()
temp_subjs = imu_subjects.iloc[temp_idx].copy()
temp_strat = strat_label.iloc[temp_idx].copy()
temp_counts = temp_strat.value_counts()
temp_small = temp_counts[temp_counts < 2].index
temp_strat = temp_strat.apply(lambda x: temp_subjs.loc[temp_strat == x, 'group'].iloc[0] if x in temp_small else x)

# Split temp 50/50 -> 15% val, 15% test
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
val_sub_idx, test_sub_idx = next(sss2.split(temp_subjs, temp_strat))

val_subjs = temp_subjs.iloc[val_sub_idx].copy()
test_subjs = temp_subjs.iloc[test_sub_idx].copy()

train_subjs['split'] = 'TRAIN'
val_subjs['split'] = 'VAL'
test_subjs['split'] = 'TEST'

imu_splits = pd.concat([train_subjs, val_subjs, test_subjs], ignore_index=True)
imu_splits_path = os.path.join(METADATA_DIR, 'imu_splits.csv')
imu_splits.to_csv(imu_splits_path, index=False)
print("=== IMU SPLITS SUMMARY ===")
print(imu_splits.groupby(['split', 'group'])['subject'].count().unstack(fill_value=0))
print(f"Total IMU subjects: {len(imu_splits)} (TRAIN: {len(train_subjs)}, VAL: {len(val_subjs)}, TEST: {len(test_subjs)})")
print(f"Saved: {imu_splits_path}")

# ==========================================
# 2. sEMG PATIENT-LEVEL SPLIT
# ==========================================
semg_summary_path = os.path.join(PROJECT_ROOT, 'data', 'real', 'semg', 'bundle', 'full_indicators_summary.csv')
df_semg = pd.read_csv(semg_summary_path, encoding='gbk')

semg_subjects = df_semg[['number', 'group', 'K-L grade', 'age', 'gender', 'BMI']].copy()
semg_subjects.rename(columns={'number': 'subject_id', 'group': 'class_label'}, inplace=True)

# Stratify by class_label (0=Healthy, 1=KOA)
sss_semg1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
train_semg_idx, temp_semg_idx = next(sss_semg1.split(semg_subjects, semg_subjects['class_label']))

train_semg = semg_subjects.iloc[train_semg_idx].copy()
temp_semg = semg_subjects.iloc[temp_semg_idx].copy()

sss_semg2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
val_semg_idx, test_semg_idx = next(sss_semg2.split(temp_semg, temp_semg['class_label']))

val_semg = temp_semg.iloc[val_semg_idx].copy()
test_semg = temp_semg.iloc[test_semg_idx].copy()

train_semg['split'] = 'TRAIN'
val_semg['split'] = 'VAL'
test_semg['split'] = 'TEST'

semg_splits = pd.concat([train_semg, val_semg, test_semg], ignore_index=True)
semg_splits_path = os.path.join(METADATA_DIR, 'semg_splits.csv')
semg_splits.to_csv(semg_splits_path, index=False)

print("\n=== sEMG SPLITS SUMMARY ===")
print(semg_splits.groupby(['split', 'class_label'])['subject_id'].count().unstack(fill_value=0))
print(f"Total sEMG subjects: {len(semg_splits)} (TRAIN: {len(train_semg)}, VAL: {len(val_semg)}, TEST: {len(test_semg)})")
print(f"Saved: {semg_splits_path}")
