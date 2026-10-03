"""
Synthetic Generator Calibration Module.
CRITICAL CONSTRAINT: Fits distributions and correlations strictly on the REAL TRAINING SPLIT.
Never exposes test set distributions.
"""
import os
import json
import numpy as np
import pandas as pd

class RealDataCalibrator:
    """
    Learns class-conditional empirical distributions, means, standard deviations,
    and covariance structures from the real training splits.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.imu_stats = {}
        self.semg_stats = {}
        self.calibrated = False

    def fit_imu(self, df_imu_train, feature_cols=None):
        """Fits distributions on real IMU training data."""
        if feature_cols is None:
            feature_cols = [
                'stride_time', 'cadence', 'gait_speed', 'stance_time', 'swing_time',
                'stride_time_variability', 'stance_phase_asymmetry', 'stride_asymmetry',
                'acceleration_mean', 'acceleration_std', 'acceleration_rms',
                'angular_velocity_mean', 'angular_velocity_std', 'angular_velocity_rms',
                'peak_acceleration', 'peak_angular_velocity', 'foot_angular_velocity_asymmetry'
            ]
        
        # Filter for only available feature columns
        cols = [c for c in feature_cols if c in df_imu_train.columns]
        
        # Split into Control (HS) and KOA (or Ortho)
        ctrl_df = df_imu_train[df_imu_train['pathology_key'] == 'HS'][cols]
        koa_df = df_imu_train[df_imu_train['pathology_key'] == 'KOA'][cols]
        if len(koa_df) == 0:
            # Fallback to general ortho if pure KOA split is small
            koa_df = df_imu_train[df_imu_train['group'] == 'ortho'][cols]

        def get_group_stats(df_group):
            clean_df = df_group.dropna()
            stats = {
                'mean': clean_df.mean().to_dict(),
                'std': clean_df.std().replace(0, 1e-4).to_dict(),
                'median': clean_df.median().to_dict(),
                'iqr': (clean_df.quantile(0.75) - clean_df.quantile(0.25)).to_dict(),
                'cov': clean_df.cov().to_dict(),
                'corr': clean_df.corr().to_dict(),
                'feature_cols': cols
            }
            return stats

        self.imu_stats = {
            'control': get_group_stats(ctrl_df),
            'koa': get_group_stats(koa_df),
            'cols': cols
        }
        return self.imu_stats

    def fit_semg(self, df_semg_train, feature_cols=None):
        """Fits distributions on real sEMG training data."""
        if feature_cols is None:
            feature_cols = [
                'vm_rms', 'vl_rms', 'rf_rms', 'bf_rms',
                'vm_median_frequency', 'vl_median_frequency', 'rf_median_frequency', 'bf_median_frequency',
                'vm_vl_rms_ratio', 'vm_vl_activation_asymmetry', 'hq_ratio',
                'vm_peak_rms', 'vl_peak_rms', 'rf_peak_rms', 'bf_peak_rms',
                'vm_rms_area', 'vl_rms_area', 'rf_rms_area', 'bf_rms_area',
                'vm_activation_duration', 'vl_activation_duration', 'rf_activation_duration', 'bf_activation_duration'
            ]
            
        cols = [c for c in feature_cols if c in df_semg_train.columns]
        
        ctrl_df = df_semg_train[df_semg_train['class_label'] == 0][cols]
        koa_df = df_semg_train[df_semg_train['class_label'] == 1][cols]
        
        def get_group_stats(df_group):
            clean_df = df_group.dropna()
            stats = {
                'mean': clean_df.mean().to_dict(),
                'std': clean_df.std().replace(0, 1e-4).to_dict(),
                'median': clean_df.median().to_dict(),
                'iqr': (clean_df.quantile(0.75) - clean_df.quantile(0.25)).to_dict(),
                'cov': clean_df.cov().to_dict(),
                'corr': clean_df.corr().to_dict(),
                'feature_cols': cols
            }
            return stats

        self.semg_stats = {
            'control': get_group_stats(ctrl_df),
            'koa': get_group_stats(koa_df),
            'cols': cols
        }
        self.calibrated = True
        return self.semg_stats

    def save_calibration(self, output_path):
        """Serializes calibration statistics."""
        data = {
            'imu_stats': self.imu_stats,
            'semg_stats': self.semg_stats,
            'random_state': self.random_state
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load_calibration(self, input_path):
        """Loads serialized calibration statistics."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.imu_stats = data['imu_stats']
        self.semg_stats = data['semg_stats']
        self.calibrated = True
