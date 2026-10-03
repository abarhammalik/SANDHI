"""
Synthetic Multimodal Cohort Generation Script.
Fits RealDataCalibrator strictly on REAL TRAINING SPLITS (zero test leakage).
Generates calibrated, paired multimodal synthetic cohort with realistic physiological
impairment, sensor quality degradation, and missing-modality masks.
Saves:
  - metadata/synthetic_calibration_stats.json
  - data/synthetic/multimodal_patients.parquet
  - data/synthetic/multimodal_audit.parquet
  - data/synthetic/samples/
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.synthesis.calibrator import RealDataCalibrator
from src.synthesis.multimodal_generator import SyntheticMultimodalGenerator

REAL_IMU_PATH = os.path.join(PROJECT_ROOT, 'data', 'features', 'real_imu_features.parquet')
REAL_SEMG_PATH = os.path.join(PROJECT_ROOT, 'data', 'features', 'real_semg_features.parquet')
SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, 'data', 'synthetic')
SAMPLES_DIR = os.path.join(SYNTHETIC_DIR, 'samples')
METADATA_DIR = os.path.join(PROJECT_ROOT, 'metadata')

os.makedirs(SYNTHETIC_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

def generate_synthetic_data(num_patients=5000, random_state=42):
    print("=" * 70)
    print("ARTHROSCAN SYNTHETIC MULTIMODAL GENERATOR")
    print("=" * 70)
    t0 = time.time()

    # 1. Load Real Features
    print(f"Loading real IMU features from: {REAL_IMU_PATH}")
    df_imu = pd.read_parquet(REAL_IMU_PATH)
    print(f"Loading real sEMG features from: {REAL_SEMG_PATH}")
    df_semg = pd.read_parquet(REAL_SEMG_PATH)

    # 2. Strict Filter to TRAINING SPLITS ONLY
    imu_train = df_imu[df_imu['split'] == 'TRAIN'].copy()
    semg_train = df_semg[df_semg['split'] == 'TRAIN'].copy()
    print(f"\n[LEAKAGE PROTECTION] Real Training Cohort Sizes:")
    print(f"  Real IMU Train rows:  {len(imu_train)} (unique subjects: {imu_train['subject_id'].nunique()})")
    print(f"  Real sEMG Train rows: {len(semg_train)} (unique subjects: {semg_train['subject_id'].nunique()})")

    # 3. Fit Calibrator
    print("\nFitting RealDataCalibrator on Real Training Distributions...")
    calibrator = RealDataCalibrator(random_state=random_state)
    calibrator.fit_imu(imu_train)
    calibrator.fit_semg(semg_train)

    calib_json = os.path.join(METADATA_DIR, 'synthetic_calibration_stats.json')
    calibrator.save_calibration(calib_json)
    print(f"Saved calibration statistics to: {calib_json}")

    # 4. Generate Cohort
    print(f"\nGenerating {num_patients} paired synthetic multimodal patients (50% KOA, 50% Control)...")
    generator = SyntheticMultimodalGenerator(calibrator=calibrator, random_state=random_state)
    df_patients, df_audits, sample_raws = generator.generate_cohort(
        num_patients=num_patients,
        koa_ratio=0.50,
        raw_sample_count=5
    )

    # 5. Save Sample Raw Waveforms
    print("\nSaving raw sample waveforms for first 5 synthetic patients...")
    for pid, r_imu in sample_raws['imu']:
        if r_imu is not None:
            r_imu.to_parquet(os.path.join(SAMPLES_DIR, f"{pid}_raw_imu.parquet"), index=False)
    for pid, r_semg in sample_raws['semg']:
        if r_semg is not None:
            r_semg.to_parquet(os.path.join(SAMPLES_DIR, f"{pid}_raw_semg.parquet"), index=False)
    print(f"Raw waveform samples saved to: {SAMPLES_DIR}")

    # 6. Save Parquet Files
    patients_parquet = os.path.join(SYNTHETIC_DIR, 'multimodal_patients.parquet')
    df_patients.to_parquet(patients_parquet, index=False)
    print(f"\nSaved synthetic multimodal patients ({df_patients.shape[0]} rows, {df_patients.shape[1]} cols) to:")
    print(f"  {patients_parquet}")

    audits_parquet = os.path.join(SYNTHETIC_DIR, 'multimodal_audit.parquet')
    df_audits.to_parquet(audits_parquet, index=False)
    print(f"Saved synthetic audit log ({df_audits.shape[0]} rows) to:")
    print(f"  {audits_parquet}")

    # 7. Modality Distribution & Missingness Audit
    print("\n" + "=" * 50)
    print("SYNTHETIC COHORT BREAKDOWN")
    print("=" * 50)
    print("Class Balance:")
    print(df_patients['class_label'].value_counts().rename({0: 'Healthy Control', 1: 'KOA'}))
    print("\nModality Availability:")
    both = (df_patients['has_imu'] == 1) & (df_patients['has_semg'] == 1)
    imu_only = (df_patients['has_imu'] == 1) & (df_patients['has_semg'] == 0)
    semg_only = (df_patients['has_imu'] == 0) & (df_patients['has_semg'] == 1)
    neither = (df_patients['has_imu'] == 0) & (df_patients['has_semg'] == 0)
    print(f"  Paired (IMU + sEMG): {both.sum()} ({both.mean()*100:.1f}%)")
    print(f"  IMU-only:           {imu_only.sum()} ({imu_only.mean()*100:.1f}%)")
    print(f"  sEMG-only:          {semg_only.sum()} ({semg_only.mean()*100:.1f}%)")
    print(f"  Neither (dropped):  {neither.sum()} ({neither.mean()*100:.1f}%)")

    # 8. Statistical Fidelity Verification (KS 2-Sample Test)
    print("\n" + "=" * 50)
    print("BIOMECHANICAL & STATISTICAL FIDELITY CHECK")
    print("=" * 50)
    print(f"{'Biomarker':<30} {'Real Mean +/- Std':<22} {'Synth Mean +/- Std':<22} {'KS-pval':<10}")
    print("-" * 84)

    test_features = [
        ('stride_time', 'imu_stride_time', imu_train, df_patients[df_patients['has_imu']==1]),
        ('cadence', 'imu_cadence', imu_train, df_patients[df_patients['has_imu']==1]),
        ('gait_speed', 'imu_gait_speed', imu_train, df_patients[df_patients['has_imu']==1]),
        ('acceleration_mean', 'imu_acceleration_mean', imu_train, df_patients[df_patients['has_imu']==1]),
        ('angular_velocity_mean', 'imu_angular_velocity_mean', imu_train, df_patients[df_patients['has_imu']==1]),
        ('vm_rms', 'semg_vm_rms', semg_train, df_patients[df_patients['has_semg']==1]),
        ('vl_rms', 'semg_vl_rms', semg_train, df_patients[df_patients['has_semg']==1]),
        ('vm_median_frequency', 'semg_vm_median_frequency', semg_train, df_patients[df_patients['has_semg']==1]),
        ('vl_median_frequency', 'semg_vl_median_frequency', semg_train, df_patients[df_patients['has_semg']==1]),
        ('vm_vl_rms_ratio', 'semg_vm_vl_rms_ratio', semg_train, df_patients[df_patients['has_semg']==1])
    ]

    for feat_real, feat_synth, r_df, s_df in test_features:
        if feat_real in r_df.columns and feat_synth in s_df.columns:
            r_vals = r_df[feat_real].dropna().to_numpy()
            s_vals = s_df[feat_synth].dropna().to_numpy()
            stat, p_val = ks_2samp(r_vals, s_vals)
            r_str = f"{r_vals.mean():.3f} +/- {r_vals.std():.3f}"
            s_str = f"{s_vals.mean():.3f} +/- {s_vals.std():.3f}"
            print(f"{feat_real:<30} {r_str:<22} {s_str:<22} {p_val:<10.4f}")

    print("-" * 84)
    print(f"Synthetic cohort generation completed in {time.time() - t0:.2f}s")
    return df_patients

if __name__ == '__main__':
    generate_synthetic_data(num_patients=5000, random_state=42)
