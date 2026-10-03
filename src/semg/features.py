"""
sEMG Feature Extraction Module.
Calculates:
  - RMS amplitude for VM (Vastus Medialis) and VL (Vastus Lateralis)
  - Median Frequency (MDF) via Welch PSD
  - Derived indices: vm_vl_rms_ratio, vm_vl_activation_asymmetry, activation_duration, peak_rms, RMS_area
  - MVC-normalized activation values
  - RF and BF features
  - Provenance: source_type, source_dataset, source_subject_id, class_label, kl_grade
"""
import numpy as np
import pandas as pd
from scipy.signal import welch

def compute_rms(signal):
    """Computes root mean square (RMS) amplitude."""
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal**2)))

def compute_median_frequency(signal, fs=2000.0, nperseg=1000):
    """
    Computes Median Frequency (MDF) using Welch power spectral density.
    MDF is the frequency dividing the power spectrum into two regions of equal cumulative energy.
    """
    if len(signal) < nperseg:
        nperseg = len(signal)
    if nperseg <= 10:
        return 70.0 # fallback nominal MDF
        
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    # Exclude DC component (freq == 0)
    freqs = freqs[1:]
    psd = psd[1:]
    
    total_power = np.sum(psd)
    if total_power <= 0 or np.isnan(total_power):
        return 70.0
        
    cum_power = np.cumsum(psd)
    idx = np.where(cum_power >= total_power / 2.0)[0]
    if len(idx) > 0:
        return float(freqs[idx[0]])
    return float(freqs[-1])

def extract_muscle_features(signal, fs=2000.0, mvc_peak_rms=None, prefix='vm'):
    """Extracts features for a single muscle channel."""
    rms = compute_rms(signal)
    mdf = compute_median_frequency(signal, fs=fs)
    
    # Envelope-based peak and area
    abs_sig = np.abs(signal)
    peak_rms = float(np.max(abs_sig)) if len(abs_sig) > 0 else 0.0
    rms_area = float(np.trapezoid(abs_sig, dx=1.0/fs)) if len(abs_sig) > 0 else 0.0
    
    # Activation thresholding: time above 20% of peak
    thresh = 0.20 * peak_rms
    active_samples = np.sum(abs_sig >= thresh)
    activation_duration = float(active_samples / fs)
    
    # MVC normalization if available
    mvc_norm = float(rms / mvc_peak_rms * 100.0) if (mvc_peak_rms and mvc_peak_rms > 0) else np.nan
    
    return {
        f'{prefix}_rms': rms,
        f'{prefix}_median_frequency': mdf,
        f'{prefix}_peak_rms': peak_rms,
        f'{prefix}_rms_area': rms_area,
        f'{prefix}_activation_duration': activation_duration,
        f'{prefix}_mvc_normalized': mvc_norm
    }

def extract_semg_trial_features(muscle_signals, mvc_peaks=None, clinical_row=None,
                                source_type='REAL', source_dataset='dryad_fn2z34v8s_quadriceps_semg'):
    """
    Extracts comprehensive feature set across VM, VL, RF, and BF muscles.
    Computes VM/VL ratios, bilateral asymmetry, and integrates clinical labels.
    """
    fs = 2000.0
    out = {}
    
    # 1. Vastus Medialis (VM) and Vastus Lateralis (VL)
    vm_sig = muscle_signals.get('vm', np.array([]))
    vl_sig = muscle_signals.get('vl', np.array([]))
    rf_sig = muscle_signals.get('rf', np.array([]))
    bf_sig = muscle_signals.get('bf', np.array([]))
    
    mvc_vm = mvc_peaks.get('vm') if mvc_peaks else None
    mvc_vl = mvc_peaks.get('vl') if mvc_peaks else None
    mvc_rf = mvc_peaks.get('rf') if mvc_peaks else None
    mvc_bf = mvc_peaks.get('bf') if mvc_peaks else None
    
    vm_feats = extract_muscle_features(vm_sig, fs=fs, mvc_peak_rms=mvc_vm, prefix='vm')
    vl_feats = extract_muscle_features(vl_sig, fs=fs, mvc_peak_rms=mvc_vl, prefix='vl')
    rf_feats = extract_muscle_features(rf_sig, fs=fs, mvc_peak_rms=mvc_rf, prefix='rf')
    bf_feats = extract_muscle_features(bf_sig, fs=fs, mvc_peak_rms=mvc_bf, prefix='bf')
    
    out.update(vm_feats)
    out.update(vl_feats)
    out.update(rf_feats)
    out.update(bf_feats)
    
    # 2. Key Derived Clinical Indices
    # VM / VL Ratio
    vl_rms_val = vl_feats['vl_rms']
    vm_rms_val = vm_feats['vm_rms']
    out['vm_vl_rms_ratio'] = float(vm_rms_val / vl_rms_val) if vl_rms_val > 1e-6 else 1.0
    
    # VM vs VL Activation Asymmetry (Quadriceps imbalance index)
    out['vm_vl_activation_asymmetry'] = float(abs(vm_rms_val - vl_rms_val) / ((vm_rms_val + vl_rms_val)/2.0 + 1e-6))
    
    # Hamstring-to-Quadriceps (H:Q) Ratio: BF / (VM + VL)
    quad_rms = vm_rms_val + vl_rms_val
    bf_rms_val = bf_feats['bf_rms']
    out['hq_ratio'] = float(bf_rms_val / quad_rms) if quad_rms > 1e-6 else 0.5
    
    # 3. Clinical metadata and Provenance
    subj_id = clinical_row.get('number', 0) if clinical_row is not None else 0
    class_label = clinical_row.get('group', 0) if clinical_row is not None else 0
    kl_grade = clinical_row.get('K-L grade', 0) if clinical_row is not None else 0
    age = clinical_row.get('age', 50) if clinical_row is not None else 50
    gender = clinical_row.get('gender', 0) if clinical_row is not None else 0
    bmi = clinical_row.get('BMI', 24.0) if clinical_row is not None else 24.0
    womac = clinical_row.get('WOMAC score', np.nan) if clinical_row is not None else np.nan
    vas = clinical_row.get('VAS score', np.nan) if clinical_row is not None else np.nan
    
    out['source_type'] = source_type
    out['source_dataset'] = source_dataset
    out['source_subject_id'] = f"real_semg_{int(subj_id):03d}" if source_type == 'REAL' else str(subj_id)
    out['subject_id'] = int(subj_id)
    out['class_label'] = int(class_label)
    out['kl_grade'] = int(kl_grade)
    out['age'] = float(age)
    out['gender'] = int(gender)
    out['bmi'] = float(bmi)
    out['womac_score'] = float(womac)
    out['vas_score'] = float(vas)
    out['semg_quality'] = 1.0 # 1.0 for real data
    
    return out
