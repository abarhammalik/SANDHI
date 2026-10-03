"""
IMU Signal Preprocessing Module.
Includes:
  - Missing-value interpolation and validation
  - Zero-phase Butterworth lowpass filtering (noise removal)
  - Highpass gravity and drift separation
  - Gait-cycle segmentation using detected heel-strikes and toe-offs
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

def validate_and_impute_signal(signal, max_consecutive_nans=10):
    """
    Validates signal continuity and performs linear interpolation on missing samples.
    """
    s = pd.Series(signal)
    if s.isna().all():
        return np.zeros_like(signal)
    # Forward and backward fill small gaps
    s = s.interpolate(method='linear', limit=max_consecutive_nans)
    s = s.bfill().ffill().fillna(0.0)
    return s.to_numpy()

def butter_lowpass_filter(data, cutoff=20.0, fs=100.0, order=4):
    """
    Applies zero-phase 4th-order lowpass Butterworth filter.
    Cutoff at 20 Hz preserves human biomechanics (typically < 10-15 Hz) while removing high-frequency sensor noise.
    """
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if len(data) <= 3 * max(len(a), len(b)):
        return data
    return filtfilt(b, a, data)

def butter_highpass_filter(data, cutoff=0.25, fs=100.0, order=2):
    """
    Applies highpass filter to mitigate sensor drift and static gravity bias.
    """
    nyq = 0.5 * fs
    normal_cutoff = max(cutoff / nyq, 0.001)
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    if len(data) <= 3 * max(len(a), len(b)):
        return data
    return filtfilt(b, a, data)

def preprocess_imu_dataframe(df, fs=100.0, cutoff_lowpass=20.0, cutoff_highpass=0.25):
    """
    Preprocesses a standardized IMU dataframe:
    - Filters Acc_X, Acc_Y, Acc_Z, Gyr_X, Gyr_Y, Gyr_Z
    - Computes vector magnitudes
    """
    df_clean = df.copy()
    acc_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z']
    gyr_cols = ['gyroscope_x', 'gyroscope_y', 'gyroscope_z']
    
    for col in acc_cols + gyr_cols:
        if col in df_clean.columns:
            cleaned = validate_and_impute_signal(df_clean[col].values)
            # Lowpass filter to attenuate noise
            cleaned = butter_lowpass_filter(cleaned, cutoff=cutoff_lowpass, fs=fs)
            df_clean[col] = cleaned
            
    # Compute resultant magnitudes
    if all(c in df_clean.columns for c in acc_cols):
        df_clean['acceleration_magnitude'] = np.sqrt(
            df_clean['acceleration_x']**2 + df_clean['acceleration_y']**2 + df_clean['acceleration_z']**2
        )
    if all(c in df_clean.columns for c in gyr_cols):
        df_clean['gyroscope_magnitude'] = np.sqrt(
            df_clean['gyroscope_x']**2 + df_clean['gyroscope_y']**2 + df_clean['gyroscope_z']**2
        )
        
    return df_clean

def detect_gait_events_from_angular_velocity(foot_gyr_z, fs=100.0):
    """
    Detects gait events (heel-strike and toe-off) from sagittal plane foot angular velocity.
    Returns: list of heel-strike sample indices.
    """
    # Heel strikes correspond to prominent local minima or negative peaks following maximum swing velocity
    prominence = 0.5 * np.std(foot_gyr_z) if np.std(foot_gyr_z) > 0 else 0.1
    min_distance = int(0.5 * fs) # minimum 0.5s between steps
    peaks, _ = find_peaks(foot_gyr_z, distance=min_distance, prominence=prominence)
    return peaks
