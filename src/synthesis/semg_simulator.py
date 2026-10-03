"""
Physiological sEMG Signal and Feature Simulator.
Synthesizes raw quadriceps sEMG waveforms (VM, VL, RF, BF) with motor unit firing bursts,
frequency spectra, and vastus balance modulation driven by real training distributions.
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from src.synthesis.quality_simulator import QualitySimulator
from src.semg.preprocessor import preprocess_raw_semg
from src.semg.features import compute_rms, compute_median_frequency

class SyntheticsEMGSimulator:
    def __init__(self, calibrator=None, random_state=42):
        self.rng = np.random.RandomState(random_state)
        self.calibrator = calibrator
        self.quality_sim = QualitySimulator(random_state=random_state)

    def generate_patient_semg_features(self, is_koa=False, latent_compensation=0.0):
        """
        Samples multivariate-correlated sEMG features from calibrated training distributions
        modulated by neuromuscular compensation.
        """
        group_key = 'koa' if is_koa else 'control'
        
        if self.calibrator and self.calibrator.semg_stats and group_key in self.calibrator.semg_stats:
            stats = self.calibrator.semg_stats[group_key]
            cols = stats['feature_cols']
            means = np.array([stats['mean'][c] for c in cols])
            cov = np.array([[stats['cov'][c1][c2] for c2 in cols] for c1 in cols])
            cov += np.eye(len(cols)) * 1e-4
            sample = self.rng.multivariate_normal(means, cov)
            feat_dict = {}
            for col, val in zip(cols, sample):
                if any(k in col for k in ['rms', 'duration', 'ratio', 'area', 'asymmetry', 'normalized']):
                    feat_dict[col] = float(max(val, 1e-4))
                elif 'frequency' in col:
                    feat_dict[col] = float(np.clip(val, 15.0, 300.0))
                else:
                    feat_dict[col] = float(val)
        else:
            # Baseline physiological ranges
            if is_koa:
                vm_rms = self.rng.normal(0.045, 0.015) - 0.015 * latent_compensation
                vl_rms = self.rng.normal(0.062, 0.018)
                vm_mdf = self.rng.normal(56.0, 7.0) - 8.0 * latent_compensation
                vl_mdf = self.rng.normal(59.0, 6.5)
                rf_rms = self.rng.normal(0.050, 0.015)
                bf_rms = self.rng.normal(0.038, 0.012)
            else:
                vm_rms = self.rng.normal(0.075, 0.018)
                vl_rms = self.rng.normal(0.072, 0.016)
                vm_mdf = self.rng.normal(72.0, 6.0)
                vl_mdf = self.rng.normal(71.0, 5.5)
                rf_rms = self.rng.normal(0.065, 0.016)
                bf_rms = self.rng.normal(0.032, 0.010)

            vm_rms = max(vm_rms, 0.005)
            vl_rms = max(vl_rms, 0.005)
            rf_rms = max(rf_rms, 0.005)
            bf_rms = max(bf_rms, 0.005)
            
            ratio = vm_rms / vl_rms
            asymmetry = abs(vm_rms - vl_rms) / ((vm_rms + vl_rms)/2.0)
            
            feat_dict = {
                'vm_rms': float(vm_rms),
                'vl_rms': float(vl_rms),
                'rf_rms': float(rf_rms),
                'bf_rms': float(bf_rms),
                'vm_median_frequency': float(np.clip(vm_mdf, 30.0, 110.0)),
                'vl_median_frequency': float(np.clip(vl_mdf, 30.0, 110.0)),
                'rf_median_frequency': float(np.clip(vm_mdf + 2.0, 30.0, 110.0)),
                'bf_median_frequency': float(np.clip(vm_mdf - 3.0, 30.0, 110.0)),
                'vm_vl_rms_ratio': float(ratio),
                'vm_vl_activation_asymmetry': float(asymmetry),
                'hq_ratio': float(bf_rms / (vm_rms + vl_rms)),
                'activation_duration': float(self.rng.normal(0.45, 0.06)),
                'peak_rms': float(max(vm_rms, vl_rms) * 2.8),
                'RMS_area': float(max(vm_rms, vl_rms) * 15.0)
            }
            
        if latent_compensation > 0:
            # KOA induces VM inhibition and relative VL dominance
            feat_dict['vm_vl_rms_ratio'] = float(feat_dict.get('vm_vl_rms_ratio', 1.0) * (1.0 - 0.25 * latent_compensation))
            feat_dict['vm_vl_activation_asymmetry'] = float(feat_dict.get('vm_vl_activation_asymmetry', 0.1) * (1.0 + 0.35 * latent_compensation))
            feat_dict['vm_median_frequency'] = float(feat_dict.get('vm_median_frequency', 70.0) * (1.0 - 0.15 * latent_compensation))

        return feat_dict

    def generate_raw_semg_bursts(self, feat_dict, duration_sec=15.0, fs=2000.0):
        """
        Synthesizes raw bipolar differential sEMG waveforms matching target RMS and MDF.
        Passes synthesized signal through the standard preprocessing pipeline.
        """
        n_samples = int(duration_sec * fs)
        t = np.arange(n_samples) / fs
        
        # Gait activation cycle (~1 Hz stride frequency)
        burst_envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 1.0 * t))**2
        
        def synth_channel(target_rms, target_mdf):
            # Generate Gaussian white noise
            raw_noise = self.rng.normal(0, 1.0, size=n_samples)
            # Bandpass shape to match target MDF
            # Resonant peak near target_mdf
            low = max(target_mdf * 0.4, 20.0)
            high = min(target_mdf * 1.8, 450.0)
            nyq = 0.5 * fs
            b, a = butter(2, [low/nyq, high/nyq], btype='band')
            filtered_noise = filtfilt(b, a, raw_noise)
            # Modulate with burst envelope
            burst_sig = filtered_noise * burst_envelope
            # Scale to target RMS
            cur_rms = compute_rms(burst_sig)
            if cur_rms > 0:
                burst_sig = burst_sig * (target_rms / cur_rms)
            # Apply quality simulation (motion artifact, 50 Hz hum)
            deg_sig, quality = self.quality_sim.degrade_semg_signal(burst_sig, fs=fs)
            return deg_sig, quality

        vm_sig, q_vm = synth_channel(feat_dict['vm_rms'], feat_dict['vm_median_frequency'])
        vl_sig, q_vl = synth_channel(feat_dict['vl_rms'], feat_dict['vl_median_frequency'])
        rf_sig, _ = synth_channel(feat_dict['rf_rms'], feat_dict['rf_median_frequency'])
        bf_sig, _ = synth_channel(feat_dict['bf_rms'], feat_dict['bf_median_frequency'])
        
        overall_quality = float(np.mean([q_vm, q_vl]))
        df_semg_raw = pd.DataFrame({
            'timestamp': t,
            'vm': vm_sig,
            'vl': vl_sig,
            'rf': rf_sig,
            'bf': bf_sig
        })
        return df_semg_raw, overall_quality
