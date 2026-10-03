"""
sEMG Processing and Feature Extraction Script.
Processes all 88 subjects from Dryad dataset (KOA vs Healthy controls).
Extracts VM, VL, RF, BF muscle activation, spectral MDF, VM/VL ratio, and clinical labels.
Saves:
  - data/features/real_semg_features.parquet
"""
import os
import sys
import time
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.semg.loader import load_clinical_summary, find_subject_file, load_subject_raw_semg
from src.semg.preprocessor import preprocess_raw_semg
from src.semg.features import extract_semg_trial_features, compute_rms

SEMG_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'real', 'semg', 'raw', 'full sEMG data in correct format')
SEMG_BUNDLE_DIR = os.path.join(PROJECT_ROOT, 'data', 'real', 'semg', 'bundle')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'features')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'semg')
METADATA_DIR = os.path.join(PROJECT_ROOT, 'metadata')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

splits_path = os.path.join(METADATA_DIR, 'semg_splits.csv')
df_splits = pd.read_csv(splits_path) if os.path.exists(splits_path) else None
split_map = dict(zip(df_splits['subject_id'], df_splits['split'])) if df_splits is not None else {}

def get_channel_signal(df, sensor_num, channel_idx=2):
    """Safely extracts signal array for a sensor and channel."""
    # Look for exact or fuzzy match
    target_names = [
        f"EMG_IMU_{sensor_num}_EMG_CH{channel_idx}",
        f"EMG_IMU_{sensor_num}_EMG_CH1",
        f"EMG_{sensor_num}_CH{channel_idx}",
        f"CH{sensor_num}"
    ]
    for name in target_names:
        if name in df.columns:
            return df[name].to_numpy(dtype=float)
    # Check if any column contains sensor_num and CH
    for col in df.columns:
        if f"_{sensor_num}_EMG" in col and f"CH{channel_idx}" in col:
            return df[col].to_numpy(dtype=float)
    return None

def extract_subject_muscle_signals(df_raw):
    """
    Extracts VM, VL, RF, and BF signals.
    Averages bilateral recordings when both left and right are available.
    """
    signals = {}
    
    # Vastus Medialis (Left=1, Right=9)
    vm_l = get_channel_signal(df_raw, 1, 2)
    vm_r = get_channel_signal(df_raw, 9, 2)
    if vm_l is not None and vm_r is not None:
        min_len = min(len(vm_l), len(vm_r))
        signals['vm'] = (vm_l[:min_len] + vm_r[:min_len]) / 2.0
    elif vm_l is not None:
        signals['vm'] = vm_l
    elif vm_r is not None:
        signals['vm'] = vm_r
    else:
        signals['vm'] = np.zeros(1000)
        
    # Vastus Lateralis (Left=3, Right=11)
    vl_l = get_channel_signal(df_raw, 3, 2)
    vl_r = get_channel_signal(df_raw, 11, 2)
    if vl_l is not None and vl_r is not None:
        min_len = min(len(vl_l), len(vl_r))
        signals['vl'] = (vl_l[:min_len] + vl_r[:min_len]) / 2.0
    elif vl_l is not None:
        signals['vl'] = vl_l
    elif vl_r is not None:
        signals['vl'] = vl_r
    else:
        signals['vl'] = np.zeros(1000)
        
    # Rectus Femoris (Left=2, Right=10)
    rf_l = get_channel_signal(df_raw, 2, 2)
    rf_r = get_channel_signal(df_raw, 10, 2)
    if rf_l is not None and rf_r is not None:
        min_len = min(len(rf_l), len(rf_r))
        signals['rf'] = (rf_l[:min_len] + rf_r[:min_len]) / 2.0
    elif rf_l is not None:
        signals['rf'] = rf_l
    elif rf_r is not None:
        signals['rf'] = rf_r
    else:
        signals['rf'] = np.zeros(1000)
        
    # Biceps Femoris (Left=4/5, Right=12/13)
    bf_l = get_channel_signal(df_raw, 4, 2)
    if bf_l is None:
        bf_l = get_channel_signal(df_raw, 5, 2)
    bf_r = get_channel_signal(df_raw, 12, 2)
    if bf_r is None:
        bf_r = get_channel_signal(df_raw, 13, 2)
    if bf_l is not None and bf_r is not None:
        min_len = min(len(bf_l), len(bf_r))
        signals['bf'] = (bf_l[:min_len] + bf_r[:min_len]) / 2.0
    elif bf_l is not None:
        signals['bf'] = bf_l
    elif bf_r is not None:
        signals['bf'] = bf_r
    else:
        signals['bf'] = np.zeros(1000)
        
    return signals

def run_semg_feature_extraction():
    print("Loading clinical summary...")
    df_clinical = load_clinical_summary(SEMG_BUNDLE_DIR)
    print(f"Total clinical subjects in summary: {len(df_clinical)}")
    
    records = []
    t0 = time.time()
    
    for _, clin_row in df_clinical.iterrows():
        sid = int(clin_row['number'])
        
        # 1. Load MVC test to get peak reference amplitudes
        mvc_file = find_subject_file(SEMG_RAW_DIR, 'mvc', sid)
        mvc_peaks = {}
        if mvc_file:
            try:
                df_mvc = load_subject_raw_semg(mvc_file)
                mvc_raw_sigs = extract_subject_muscle_signals(df_mvc)
                for m, sig in mvc_raw_sigs.items():
                    filt, rect, env = preprocess_raw_semg(sig)
                    mvc_peaks[m] = float(np.max(env)) if len(env) > 0 else 1.0
            except Exception as e:
                print(f"  Subject {sid} MVC load warning: {e}")
                
        # 2. Load Walk test (primary functional gait task)
        walk_file = find_subject_file(SEMG_RAW_DIR, 'walk', sid)
        if not walk_file:
            print(f"  Missing walk file for subject {sid}")
            continue
            
        try:
            df_walk = load_subject_raw_semg(walk_file)
            walk_raw_sigs = extract_subject_muscle_signals(df_walk)
            
            # Preprocess walk signals
            walk_proc_sigs = {}
            for m, sig in walk_raw_sigs.items():
                filt, rect, env = preprocess_raw_semg(sig)
                walk_proc_sigs[m] = filt # use filtered bipolar signal for RMS and MDF
                
            # Extract features
            feats = extract_semg_trial_features(
                walk_proc_sigs,
                mvc_peaks=mvc_peaks,
                clinical_row=clin_row,
                source_type='REAL',
                source_dataset='dryad_fn2z34v8s_quadriceps_semg'
            )
            
            feats['split'] = split_map.get(sid, 'TRAIN')
            feats['task'] = 'walk'
            
            # Save first 5 preprocessed signals to data/processed/semg/
            if sid <= 5:
                sample_proc_path = os.path.join(PROCESSED_DIR, f"subject_{sid:03d}_walk_preprocessed.parquet")
                pd.DataFrame(walk_proc_sigs).to_parquet(sample_proc_path, index=False)
                
            records.append(feats)
        except Exception as e:
            print(f"  Error processing subject {sid} walk sEMG: {e}")
            
    df_feats = pd.DataFrame(records)
    out_parquet = os.path.join(OUTPUT_DIR, 'real_semg_features.parquet')
    df_feats.to_parquet(out_parquet, index=False)
    print(f"\nSaved sEMG features ({df_feats.shape[0]} rows, {df_feats.shape[1]} cols) to {out_parquet}")
    if len(df_feats) > 0 and 'split' in df_feats.columns:
        print("\nClass breakdown by split:")
        print(df_feats.groupby(['split', 'class_label'])['subject_id'].count().unstack(fill_value=0))
    else:
        print("\nWARNING: No sEMG features extracted — check errors above.")
    print(f"Completed sEMG extraction in {time.time() - t0:.2f}s")
    return df_feats

if __name__ == '__main__':
    run_semg_feature_extraction()
