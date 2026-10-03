"""
Standardized IMU Data Loader.
Parses raw 100 Hz sensor files (_HE, _LB, _LF, _RF) and trial metadata.
Preserves all raw channels and provides a unified schema.
"""
import os
import glob
import json
import pandas as pd
import numpy as np

SENSOR_MAP = {
    'HE': 'head',
    'LB': 'lower_back',
    'LF': 'left_foot',
    'RF': 'right_foot'
}

def load_trial_metadata(trial_dir):
    """Loads metadata JSON from a trial directory."""
    meta_files = [f for f in os.listdir(trial_dir) if f.endswith('_meta.json')]
    if not meta_files:
        raise FileNotFoundError(f"No metadata JSON found in {trial_dir}")
    meta_path = os.path.join(trial_dir, meta_files[0])
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return meta

def load_raw_sensors(trial_dir):
    """
    Loads all available raw sensor files (_HE, _LB, _LF, _RF) in a trial directory.
    Returns dict: {sensor_code: DataFrame}
    """
    raw_data = {}
    for code in SENSOR_MAP.keys():
        sensor_file = None
        for f in os.listdir(trial_dir):
            if f.endswith(f"_{code}.txt") and not 'processed' in f:
                sensor_file = os.path.join(trial_dir, f)
                break
        if sensor_file and os.path.exists(sensor_file):
            df = pd.read_csv(sensor_file, sep='\t')
            raw_data[code] = df
        else:
            raw_data[code] = None
    return raw_data

def load_processed_file(trial_dir):
    """Loads processed synchronized data table if present."""
    proc_files = [f for f in os.listdir(trial_dir) if 'processed' in f and f.endswith('.txt')]
    if not proc_files:
        return None
    proc_path = os.path.join(trial_dir, proc_files[0])
    return pd.read_csv(proc_path, sep='\t')

def create_standardized_imu_table(trial_dir):
    """
    Constructs a standardized representation from trial files:
    Columns:
      timestamp, subject_id, trial_id, session_id, sensor_id, sensor_location,
      acceleration_x, acceleration_y, acceleration_z,
      gyroscope_x, gyroscope_y, gyroscope_z,
      magnetometer_x, magnetometer_y, magnetometer_z
    """
    meta = load_trial_metadata(trial_dir)
    raw_sensors = load_raw_sensors(trial_dir)
    
    subject_id = meta.get('subject', 'unknown')
    trial_id = meta.get('trial', 1)
    session_id = meta.get('session', 1)
    freq = float(meta.get('freq', 100.0))
    dt = 1.0 / freq
    
    records = []
    for code, df in raw_sensors.items():
        if df is None or len(df) == 0:
            continue
        location = SENSOR_MAP[code]
        n_samples = len(df)
        time_arr = np.arange(n_samples) * dt
        
        acc_x = df['Acc_X'].to_numpy() if 'Acc_X' in df.columns else np.zeros(n_samples)
        acc_y = df['Acc_Y'].to_numpy() if 'Acc_Y' in df.columns else np.zeros(n_samples)
        acc_z = df['Acc_Z'].to_numpy() if 'Acc_Z' in df.columns else np.zeros(n_samples)
        
        gyr_x = df['Gyr_X'].to_numpy() if 'Gyr_X' in df.columns else np.zeros(n_samples)
        gyr_y = df['Gyr_Y'].to_numpy() if 'Gyr_Y' in df.columns else np.zeros(n_samples)
        gyr_z = df['Gyr_Z'].to_numpy() if 'Gyr_Z' in df.columns else np.zeros(n_samples)
        
        mag_x = df['Mag_X'].to_numpy() if 'Mag_X' in df.columns else np.zeros(n_samples)
        mag_y = df['Mag_Y'].to_numpy() if 'Mag_Y' in df.columns else np.zeros(n_samples)
        mag_z = df['Mag_Z'].to_numpy() if 'Mag_Z' in df.columns else np.zeros(n_samples)
        
        sub_df = pd.DataFrame({
            'timestamp': time_arr,
            'subject_id': subject_id,
            'trial_id': trial_id,
            'session_id': session_id,
            'sensor_id': code,
            'sensor_location': location,
            'acceleration_x': acc_x,
            'acceleration_y': acc_y,
            'acceleration_z': acc_z,
            'gyroscope_x': gyr_x,
            'gyroscope_y': gyr_y,
            'gyroscope_z': gyr_z,
            'magnetometer_x': mag_x,
            'magnetometer_y': mag_y,
            'magnetometer_z': mag_z
        })
        records.append(sub_df)
        
    if not records:
        return pd.DataFrame()
        
    return pd.concat(records, ignore_index=True)
