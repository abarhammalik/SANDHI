"""
ARTHROSCAN Clinical Inference & Testing Script.
Allows testing trained models on:
  1. Random or specific real test patients from the holdout set
  2. IMU-only patient records
  3. sEMG-only patient records
  4. Fully paired multimodal records
Outputs risk scores, classification labels, and clinical interpretations.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'early_fusion_best.joblib')
DATASET_PATH = os.path.join(PROJECT_ROOT, 'data', 'features', 'multimodal_fusion_dataset.parquet')

def load_inference_pipeline():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run scripts/train_and_evaluate.py first.")
    return joblib.load(MODEL_PATH)

def test_samples(num_samples=5, sample_type='random'):
    model = load_inference_pipeline()
    df = pd.read_parquet(DATASET_PATH)

    # Filter to real holdout test set with binary labels
    test_df = df[(df['split'] == 'TEST') & (df['source_type'] == 'REAL') & (df['class_label'].isin([0, 1]))].copy()

    if sample_type == 'imu_only':
        subset = test_df[(test_df['has_imu'] == 1) & (test_df['has_semg'] == 0)]
    elif sample_type == 'semg_only':
        subset = test_df[(test_df['has_imu'] == 0) & (test_df['has_semg'] == 1)]
    elif sample_type == 'paired':
        # Simulated paired test from synth
        synth_df = df[(df['source_type'] == 'SYNTHETIC') & (df['has_imu'] == 1) & (df['has_semg'] == 1)]
        subset = synth_df
    else:
        subset = test_df

    n = min(num_samples, len(subset))
    samples = subset.sample(n=n, random_state=np.random.randint(1000))

    # Feature columns matching training
    imu_feature_cols = [c for c in df.columns if c.startswith('imu_') and c not in ['imu_mask', 'imu_quality']]
    semg_feature_cols = [c for c in df.columns if c.startswith('semg_') and c not in ['semg_mask', 'semg_quality']]
    demo_cols = ['age', 'gender', 'bmi']
    meta_flag_cols = ['imu_mask', 'semg_mask', 'imu_quality', 'semg_quality']
    all_feature_cols = imu_feature_cols + semg_feature_cols + demo_cols + meta_flag_cols

    X = samples[all_feature_cols]
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    print("=" * 85)
    print(f"ARTHROSCAN MODEL TESTING (Cohort: {sample_type.upper()}, N={n})")
    print("=" * 85)
    print(f"{'Patient ID':<22} {'True Label':<12} {'Predicted':<12} {'KOA Risk':<10} {'Modalities':<14} {'Result':<8}")
    print("-" * 85)

    for i, (_, row) in enumerate(samples.iterrows()):
        pid = str(row['patient_id'])
        true_lbl = "KOA (1)" if row['class_label'] == 1 else "Healthy (0)"
        pred_lbl = "KOA (1)" if preds[i] == 1 else "Healthy (0)"
        risk = f"{probs[i]*100:.1f}%"
        
        mods = []
        if row['has_imu'] == 1:
            mods.append("IMU")
        if row['has_semg'] == 1:
            mods.append("sEMG")
        mod_str = "+".join(mods) if mods else "None"
        
        is_correct = "CORRECT" if row['class_label'] == preds[i] else "MISMATCH"
        print(f"{pid:<22} {true_lbl:<12} {pred_lbl:<12} {risk:<10} {mod_str:<14} {is_correct:<8}")

    print("-" * 85)
    return samples, probs, preds

if __name__ == '__main__':
    # Test across different modality scenarios
    print("\n--- 1. Testing Real Holdout Patients (Mixed Modalities) ---")
    test_samples(num_samples=5, sample_type='random')

    print("\n--- 2. Testing Real IMU-Only Holdout Patients ---")
    test_samples(num_samples=3, sample_type='imu_only')

    print("\n--- 3. Testing Real sEMG-Only Holdout Patients ---")
    test_samples(num_samples=3, sample_type='semg_only')
