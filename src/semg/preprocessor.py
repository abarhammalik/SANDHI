"""
sEMG Signal Preprocessing Module.
Implements:
  - 4th-order zero-phase Butterworth bandpass filter (20 - 450 Hz)
  - 50 Hz notch filter (Q=30) to eliminate power-line interference
  - Full-wave rectification (|x(t)|)
  - Linear envelope smoothing (6 Hz lowpass)
  - MVC normalization
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def butter_bandpass_filter(data, lowcut=20.0, highcut=450.0, fs=2000.0, order=4):
    """Zero-phase Butterworth bandpass filter (20 - 450 Hz)."""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    if len(data) <= 3 * max(len(a), len(b)):
        return data
    return filtfilt(b, a, data)

def notch_filter(data, freq=50.0, Q=30.0, fs=2000.0):
    """Zero-phase IIR notch filter for 50 Hz powerline hum."""
    b, a = iirnotch(freq, Q, fs)
    if len(data) <= 3 * max(len(a), len(b)):
        return data
    return filtfilt(b, a, data)

def full_wave_rectify(data):
    """Rectification of sEMG signal."""
    return np.abs(data)

def compute_linear_envelope(rectified_data, cutoff=6.0, fs=2000.0, order=4):
    """Computes muscle activation linear envelope via 6 Hz lowpass filter."""
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.999)
    b, a = butter(order, normal_cutoff, btype='low')
    if len(rectified_data) <= 3 * max(len(a), len(b)):
        return rectified_data
    return np.maximum(filtfilt(b, a, rectified_data), 0.0)

def preprocess_raw_semg(signal, fs=2000.0, lowcut=20.0, highcut=450.0, notch_hz=50.0):
    """
    Complete standard sEMG preprocessing pipeline:
    1. Bandpass filter (20 - 450 Hz)
    2. Notch filter (50 Hz)
    3. Full-wave rectification
    4. Envelope calculation (6 Hz lowpass)
    Returns: (filtered_signal, rectified_signal, envelope)
    """
    sig = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    # Remove DC offset
    sig = sig - np.mean(sig)
    
    # 1. Bandpass
    filtered = butter_bandpass_filter(sig, lowcut=lowcut, highcut=highcut, fs=fs)
    # 2. Notch
    filtered = notch_filter(filtered, freq=notch_hz, Q=30.0, fs=fs)
    # 3. Rectify
    rectified = full_wave_rectify(filtered)
    # 4. Envelope
    envelope = compute_linear_envelope(rectified, cutoff=6.0, fs=fs)
    
    return filtered, rectified, envelope
