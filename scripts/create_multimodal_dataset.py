"""
Unified Multimodal Fusion Dataset Construction Script.
Combines:
  1. Real IMU Dataset (real gait kinematic trials with patient-level splits)
  2. Real sEMG Dataset (real quadriceps sEMG trials with patient-level splits)
  3. Calibrated Multimodal Synthetic Cohort (paired IMU+sEMG & missing modality augmentations)
Guarantees:
  - ZERO synthetic data in VAL and TEST sets (100% clean real holdout validation)
  - Unified feature schema with imu_ and semg_ prefixes
  - Explicit modality indicator masks (imu_mask, semg_mask) and quality scores (imu_quality, semg_quality)
Saves:
  - data/features/multimodal_fusion_dataset.parquet
"""
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

REAL_IMU_PATH = os.path.join(PROJECT_ROOT, 'data', 'features', 'real_imu_features.parquet')
REAL_SEMG_PATH = os.path.join(PROJECT_ROOT, 'data', 'features', 'real_semg_features.parquet')
SYNTH_PATH = os.path.join(PROJECT_ROOT, 'data', 'synthetic', 'multimodal_patients.parquet')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'features', 'multimodal_fusion_dataset.parquet')

def build_multimodal_fusion_dataset():
    print("=" * 70)
    print("BUILDING UNIFIED ARTHROSCAN MULTIMODAL FUSION DATASET")
    print("=" * 70)

    # 1. Load Datasets
    print(f"Loading Real IMU: {REAL_IMU_PATH}")
    df_imu = pd.read_parquet(REAL_IMU_PATH)
    print(f"  IMU shape: {df_imu.shape}")

    print(f"Loading Real sEMG: {REAL_SEMG_PATH}")
    df_semg = pd.read_parquet(REAL_SEMG_PATH)
    print(f"  sEMG shape: {df_semg.shape}")

    print(f"Loading Synthetic Multimodal Cohort: {SYNTH_PATH}")
    df_synth = pd.read_parquet(SYNTH_PATH)
    print(f"  Synthetic shape: {df_synth.shape}")

    # Standard feature column lists
    imu_feats = [
        'stride_time', 'cadence', 'gait_speed', 'stance_time', 'swing_time',
        'stride_time_variability', 'stance_phase_asymmetry', 'stride_asymmetry',
        'acceleration_mean', 'acceleration_std', 'acceleration_rms',
        'angular_velocity_mean', 'angular_velocity_std', 'angular_velocity_rms',
        'peak_acceleration', 'peak_angular_velocity', 'foot_angular_velocity_asymmetry'
    ]

    semg_feats = [
        'vm_rms', 'vl_rms', 'rf_rms', 'bf_rms',
        'vm_median_frequency', 'vl_median_frequency', 'rf_median_frequency', 'bf_median_frequency',
        'vm_vl_rms_ratio', 'vm_vl_activation_asymmetry', 'hq_ratio',
        'vm_peak_rms', 'vl_peak_rms', 'rf_peak_rms', 'bf_peak_rms',
        'vm_rms_area', 'vl_rms_area', 'rf_rms_area', 'bf_rms_area',
        'vm_activation_duration', 'vl_activation_duration', 'rf_activation_duration', 'bf_activation_duration'
    ]

    def parse_gender(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().upper()
        if s in ['M', 'MALE', '1', '1.0']:
            return 1.0
        if s in ['F', 'FEMALE', '0', '0.0']:
            return 0.0
        return np.nan

    # 2. Format Real IMU Records
    print("\nFormatting Real IMU records...")
    imu_records = []
    for _, row in df_imu.iterrows():
        rec = {
            'patient_id': f"real_imu_{row['source_subject_id']}",
            'source_type': 'REAL',
            'source_dataset': row.get('source_dataset', 'fukuchi_gait_imu'),
            'source_subject_id': str(row['source_subject_id']),
            'trial_id': row.get('source_trial_id', 'trial_1'),
            'split': row['split'],
            'class_label': int(row['is_koa']),
            'is_koa': int(row['is_koa']),
            'age': float(row['age']) if pd.notna(row.get('age')) else np.nan,
            'gender': parse_gender(row.get('gender')),
            'bmi': float(row['bmi']) if pd.notna(row.get('bmi')) else np.nan,
            'kl_grade': np.nan,
            
            # Modality Flags
            'has_imu': 1,
            'has_semg': 0,
            'imu_mask': 1,
            'semg_mask': 0,
            'imu_quality': float(row.get('imu_quality', 1.0)),
            'semg_quality': 0.0
        }
        for f in imu_feats:
            rec[f"imu_{f}"] = float(row[f]) if f in row and pd.notna(row[f]) else np.nan
        for f in semg_feats:
            rec[f"semg_{f}"] = np.nan
        imu_records.append(rec)
    df_imu_norm = pd.DataFrame(imu_records)

    # 3. Format Real sEMG Records
    print("Formatting Real sEMG records...")
    semg_records = []
    for _, row in df_semg.iterrows():
        rec = {
            'patient_id': f"real_semg_{row['source_subject_id']}",
            'source_type': 'REAL',
            'source_dataset': row.get('source_dataset', 'dryad_fn2z34v8s_quadriceps_semg'),
            'source_subject_id': str(row['source_subject_id']),
            'trial_id': row.get('task', 'walk'),
            'split': row['split'],
            'class_label': int(row['class_label']),
            'is_koa': int(row['class_label']),
            'age': float(row['age']) if pd.notna(row.get('age')) else np.nan,
            'gender': parse_gender(row.get('gender')),
            'bmi': float(row['bmi']) if pd.notna(row.get('bmi')) else np.nan,
            'kl_grade': float(row['kl_grade']) if pd.notna(row.get('kl_grade')) else np.nan,
            
            # Modality Flags
            'has_imu': 0,
            'has_semg': 1,
            'imu_mask': 0,
            'semg_mask': 1,
            'imu_quality': 0.0,
            'semg_quality': float(row.get('semg_quality', 1.0))
        }
        for f in imu_feats:
            rec[f"imu_{f}"] = np.nan
        for f in semg_feats:
            rec[f"semg_{f}"] = float(row[f]) if f in row and pd.notna(row[f]) else np.nan
        semg_records.append(rec)
    df_semg_norm = pd.DataFrame(semg_records)

    # 4. Format Synthetic Multimodal Records
    print("Formatting Synthetic Multimodal records...")
    df_synth_norm = df_synth.copy()
    df_synth_norm['patient_id'] = df_synth_norm['synthetic_patient_id']
    df_synth_norm['trial_id'] = 'synthetic_sim_01'
    df_synth_norm['split'] = 'TRAIN'  # Enforce synthetic in TRAIN split only

    # Align column sets
    all_cols = [
        'patient_id', 'source_type', 'source_dataset', 'source_subject_id', 'trial_id',
        'split', 'class_label', 'is_koa', 'age', 'gender', 'bmi', 'kl_grade',
        'has_imu', 'has_semg', 'imu_mask', 'semg_mask', 'imu_quality', 'semg_quality'
    ] + [f"imu_{f}" for f in imu_feats] + [f"semg_{f}" for f in semg_feats]

    df_imu_norm = df_imu_norm.reindex(columns=all_cols)
    df_semg_norm = df_semg_norm.reindex(columns=all_cols)
    df_synth_norm = df_synth_norm.reindex(columns=all_cols)

    # 5. Concatenate Unified Multimodal Dataset
    df_multimodal = pd.concat([df_imu_norm, df_semg_norm, df_synth_norm], ignore_index=True)
    str_cols = ['patient_id', 'source_type', 'source_dataset', 'source_subject_id', 'trial_id', 'split']
    for c in str_cols:
        df_multimodal[c] = df_multimodal[c].astype(str)

    df_multimodal.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved Unified Multimodal Fusion Dataset to:\n  {OUTPUT_PATH}")
    print(f"Total Rows: {len(df_multimodal)}, Total Cols: {len(df_multimodal.columns)}")

    # 6. Critical Leakage and Integrity Verification
    print("\n" + "=" * 50)
    print("DATASET INTEGRITY & SPLIT LEAKAGE AUDIT")
    print("=" * 50)
    print("1. Split Breakdown by Source Type:")
    split_summary = df_multimodal.groupby(['split', 'source_type', 'has_imu', 'has_semg'])['patient_id'].count().unstack(level=[2, 3], fill_value=0)
    print(split_summary)

    print("\n2. Contamination Check:")
    val_synth = len(df_multimodal[(df_multimodal['split'] == 'VAL') & (df_multimodal['source_type'] == 'SYNTHETIC')])
    test_synth = len(df_multimodal[(df_multimodal['split'] == 'TEST') & (df_multimodal['source_type'] == 'SYNTHETIC')])
    print(f"  Synthetic rows in VAL:  {val_synth} (Expected: 0) -> {'PASS' if val_synth == 0 else 'FAIL'}")
    print(f"  Synthetic rows in TEST: {test_synth} (Expected: 0) -> {'PASS' if test_synth == 0 else 'FAIL'}")

    print("\n3. Class Distribution Across Splits:")
    class_split = df_multimodal.groupby(['split', 'class_label'])['patient_id'].count().unstack(fill_value=0)
    print(class_split)

    return df_multimodal

if __name__ == '__main__':
    build_multimodal_fusion_dataset()
