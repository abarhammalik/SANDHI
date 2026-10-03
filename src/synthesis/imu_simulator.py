"""
Biomechanical IMU Signal and Feature Simulator.
Generates realistic cyclical gait inertial waveforms (acceleration and angular velocity)
driven by calibrated training distributions and latent biomechanical impairment factors.
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from src.synthesis.quality_simulator import QualitySimulator

class SyntheticIMUSimulator:
    def __init__(self, calibrator=None, random_state=42):
        self.rng = np.random.RandomState(random_state)
        self.calibrator = calibrator
        self.quality_sim = QualitySimulator(random_state=random_state)

    def generate_patient_gait_features(self, is_koa=False, latent_impairment=0.0):
        """
        Samples multivariate-correlated IMU features from the calibrated training distribution
        modulated by the hidden latent impairment factor.
        """
        group_key = 'koa' if is_koa else 'control'
        
        if self.calibrator and self.calibrator.imu_stats and group_key in self.calibrator.imu_stats:
            stats = self.calibrator.imu_stats[group_key]
            cols = stats['feature_cols']
            means = np.array([stats['mean'][c] for c in cols])
            cov = np.array([[stats['cov'][c1][c2] for c2 in cols] for c1 in cols])
            
            # Ensure cov is positive semi-definite
            cov += np.eye(len(cols)) * 1e-4
            sample = self.rng.multivariate_normal(means, cov)
            
            feat_dict = {}
            for col, val in zip(cols, sample):
                if any(k in col for k in ['time', 'cadence', 'speed', 'std', 'rms', 'asymmetry', 'variability']):
                    feat_dict[col] = float(max(val, 1e-4))
                else:
                    feat_dict[col] = float(val)
        else:
            # Mechanistically modeled parameters if calibration stats not yet loaded
            if is_koa:
                cadence = self.rng.normal(92.0, 8.0) - 15.0 * latent_impairment
                stride_time = 120.0 / max(cadence, 60.0)
                stride_var = self.rng.normal(0.065, 0.015) + 0.03 * latent_impairment
                asymmetry = self.rng.normal(0.12, 0.04) + 0.08 * latent_impairment
                acc_mean = self.rng.normal(9.6, 0.6)
                acc_std = self.rng.normal(2.1, 0.4)
                gyr_mean = self.rng.normal(0.95, 0.20)
                gyr_std = self.rng.normal(1.10, 0.25)
            else:
                cadence = self.rng.normal(112.0, 7.0)
                stride_time = 120.0 / cadence
                stride_var = self.rng.normal(0.032, 0.008)
                asymmetry = self.rng.normal(0.035, 0.012)
                acc_mean = self.rng.normal(9.81, 0.4)
                acc_std = self.rng.normal(2.8, 0.5)
                gyr_mean = self.rng.normal(1.25, 0.22)
                gyr_std = self.rng.normal(1.45, 0.28)

            feat_dict = {
                'stride_time': float(np.clip(stride_time, 0.7, 2.2)),
                'cadence': float(np.clip(cadence, 60.0, 140.0)),
                'gait_speed': float(np.clip(0.70 * (cadence / 60.0), 0.5, 2.0)),
                'stance_time': float(stride_time * 0.60),
                'swing_time': float(stride_time * 0.40),
                'stride_time_variability': float(np.clip(stride_var, 0.01, 0.20)),
                'stance_phase_asymmetry': float(np.clip(asymmetry * 1.2, 0.01, 0.35)),
                'stride_asymmetry': float(np.clip(asymmetry, 0.01, 0.30)),
                'acceleration_mean': float(acc_mean),
                'acceleration_std': float(acc_std),
                'acceleration_rms': float(np.sqrt(acc_mean**2 + acc_std**2)),
                'angular_velocity_mean': float(gyr_mean),
                'angular_velocity_std': float(gyr_std),
                'angular_velocity_rms': float(np.sqrt(gyr_mean**2 + gyr_std**2)),
                'peak_acceleration': float(acc_mean + 2.5 * acc_std),
                'peak_angular_velocity': float(gyr_mean + 2.8 * gyr_std),
                'foot_angular_velocity_asymmetry': float(np.clip(asymmetry * 1.1, 0.01, 0.35))
            }
            
        # Apply latent impairment influence
        if latent_impairment > 0:
            feat_dict['stride_time_variability'] = float(feat_dict.get('stride_time_variability', 0.04) * (1.0 + 0.35 * latent_impairment))
            feat_dict['stride_asymmetry'] = float(feat_dict.get('stride_asymmetry', 0.03) * (1.0 + 0.45 * latent_impairment))
            feat_dict['cadence'] = float(feat_dict.get('cadence', 105.0) * (1.0 - 0.15 * latent_impairment))

        return feat_dict

    def generate_raw_gait_waveforms(self, feat_dict, duration_sec=15.0, fs=100.0):
        """
        Synthesizes realistic time-series acceleration and angular velocity waveforms
        corresponding to the kinematic features.
        """
        n_samples = int(duration_sec * fs)
        t = np.arange(n_samples) / fs
        stride_time = feat_dict.get('stride_time', 1.15)
        stride_freq = 1.0 / stride_time
        
        # Biomechanical harmonic series for gait cycle
        # Vertical acceleration: fundamental at step frequency (2 * stride_freq)
        acc_v = 9.81 + 1.8 * np.sin(2 * np.pi * 2 * stride_freq * t) + \
                0.8 * np.sin(2 * np.pi * 4 * stride_freq * t + 0.5)
                
        # Anterior-posterior acceleration: braking and propulsion
        acc_ap = 1.2 * np.sin(2 * np.pi * stride_freq * t) + \
                 0.5 * np.sin(2 * np.pi * 2 * stride_freq * t + 1.2)
                 
        # Medio-lateral acceleration
        acc_ml = 0.6 * np.sin(2 * np.pi * stride_freq * t + 0.8)
        
        # Sagittal angular velocity (dominant rotation during foot swing)
        gyr_s = 1.8 * np.sin(2 * np.pi * stride_freq * t) + \
                0.9 * np.sin(2 * np.pi * 2 * stride_freq * t + 0.4)
        gyr_ml = 0.4 * np.sin(2 * np.pi * stride_freq * t + 0.2)
        gyr_v = 0.3 * np.sin(2 * np.pi * stride_freq * t + 0.9)
        
        acc_mag = np.sqrt(acc_v**2 + acc_ap**2 + acc_ml**2)
        gyr_mag = np.sqrt(gyr_s**2 + gyr_ml**2 + gyr_v**2)
        
        # Add realistic artifacts and quality simulation
        deg_acc, deg_gyr, quality = self.quality_sim.degrade_imu_signal(acc_mag, gyr_mag, fs=fs)
        
        df_raw = pd.DataFrame({
            'timestamp': t,
            'acceleration_magnitude': deg_acc,
            'gyroscope_magnitude': deg_gyr,
            'acceleration_x': acc_ml,
            'acceleration_y': acc_ap,
            'acceleration_z': acc_v,
            'gyroscope_x': gyr_ml,
            'gyroscope_y': gyr_s,
            'gyroscope_z': gyr_v
        })
        return df_raw, quality
