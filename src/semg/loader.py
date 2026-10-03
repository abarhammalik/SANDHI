"""
Standardized sEMG Data Loader.
Handles both .csv and .xlsx files across tasks (walk, stair, MVC).
Extracts bipolar differential sEMG channels (VM, VL, RF, BF) and clinical summary ground truth.
"""
import os
import glob
import re
import pandas as pd
import numpy as np

TASK_FOLDERS = {
    'walk': 'walk sEMG data summary',
    'stair': 'up-down stair sEMG data summary',
    'mvc': 'MVC test sEMG data summary'
}

# Key muscle channels per Dryad documentation: Channel 2 is the primary signal
MUSCLE_CHANNELS = {
    'left_vm': 'EMG_IMU_1_EMG_CH2',
    'left_rf': 'EMG_IMU_2_EMG_CH2',
    'left_vl': 'EMG_IMU_3_EMG_CH2',
    'left_bf': 'EMG_IMU_4_EMG_CH2',
    'right_vm': 'EMG_IMU_9_EMG_CH2',
    'right_rf': 'EMG_IMU_10_EMG_CH2',
    'right_vl': 'EMG_IMU_11_EMG_CH2',
    'right_bf': 'EMG_IMU_12_EMG_CH2'
}

def load_clinical_summary(bundle_dir):
    """Loads the comprehensive clinical summary from full_indicators_summary.csv."""
    summary_path = os.path.join(bundle_dir, 'full_indicators_summary.csv')
    df = pd.read_csv(summary_path, encoding='gbk')
    return df

def find_subject_file(raw_dir, task_key, subject_id):
    """Finds either the .csv or .xlsx file corresponding to subject_id in the task folder."""
    folder_name = TASK_FOLDERS[task_key]
    folder_path = os.path.join(raw_dir, folder_name)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
        
    for fname in os.listdir(folder_path):
        m = re.search(r'(\d+)', fname)
        if m and int(m.group(1)) == int(subject_id):
            return os.path.join(folder_path, fname)
    return None

def load_subject_raw_semg(file_path):
    """
    Loads raw sEMG data table from CSV or XLSX.
    Returns DataFrame containing sEMG channels.
    """
    if not file_path or not os.path.exists(file_path):
        return None
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext == '.xlsx':
        # XLSX files can have 102 columns including IMU channels; we only need sEMG columns
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    return df
