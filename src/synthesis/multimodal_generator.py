"""
Synthetic Multimodal Patient Generator.
Simulates paired IMU + sEMG records for synthetic patients sharing underlying
latent physiological impairment factors.
Enforces strict provenance, quality degradation, and missing-modality masks.
"""
import os
import json
import numpy as np
import pandas as pd
from src.synthesis.imu_simulator import SyntheticIMUSimulator
from src.synthesis.semg_simulator import SyntheticsEMGSimulator
from src.synthesis.quality_simulator import QualitySimulator

class SyntheticMultimodalGenerator:
    def __init__(self, calibrator=None, random_state=42):
        self.rng = np.random.RandomState(random_state)
        self.calibrator = calibrator
        self.imu_sim = SyntheticIMUSimulator(calibrator=calibrator, random_state=random_state)
        self.semg_sim = SyntheticsEMGSimulator(calibrator=calibrator, random_state=random_state)
        self.quality_sim = QualitySimulator(random_state=random_state)

    def generate_single_patient(self, patient_idx, is_koa=False, generate_raw_waveforms=False):
        """
        Generates a paired synthetic patient with shared latent drivers.
        """
        pid = f"synthetic_patient_{patient_idx:06d}"
        synthetic_class = 1 if is_koa else 0
        
        # 1. Sample Hidden Latent Factors (Physiological Drivers)
        # These are internal simulation drivers; NEVER ML inputs!
        if is_koa:
            latent_func = float(np.clip(self.rng.normal(0.65, 0.15), 0.20, 1.0))
            latent_mech = float(np.clip(self.rng.normal(0.60, 0.18), 0.15, 1.0))
            latent_neuro = float(np.clip(self.rng.normal(0.58, 0.16), 0.20, 1.0))
            sim_age = float(np.clip(self.rng.normal(68.0, 7.0), 45.0, 88.0))
            sim_bmi = float(np.clip(self.rng.normal(27.8, 3.8), 20.0, 42.0))
            sim_kl = int(self.rng.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15]))
        else:
            latent_func = float(np.clip(self.rng.normal(0.12, 0.08), 0.0, 0.35))
            latent_mech = float(np.clip(self.rng.normal(0.10, 0.06), 0.0, 0.30))
            latent_neuro = float(np.clip(self.rng.normal(0.10, 0.07), 0.0, 0.30))
            sim_age = float(np.clip(self.rng.normal(55.0, 10.0), 25.0, 80.0))
            sim_bmi = float(np.clip(self.rng.normal(23.5, 2.5), 18.5, 32.0))
            sim_kl = 0

        sim_gender = int(self.rng.choice([0, 1], p=[0.55, 0.45]))
        
        # 2. Generate Synthetic IMU features
        imu_feats = self.imu_sim.generate_patient_gait_features(
            is_koa=is_koa,
            latent_impairment=(latent_func + latent_mech)/2.0
        )
        
        # 3. Generate Synthetic sEMG features
        semg_feats = self.semg_sim.generate_patient_semg_features(
            is_koa=is_koa,
            latent_compensation=latent_neuro
        )
        
        # 4. Modality Quality and Missingness Simulation
        has_imu, has_semg = self.quality_sim.generate_missingness_masks()
        q_imu = float(np.clip(1.0 - 0.25 * self.rng.rand(), 0.3, 1.0)) if has_imu else 0.0
        q_semg = float(np.clip(1.0 - 0.25 * self.rng.rand(), 0.3, 1.0)) if has_semg else 0.0
        
        # 5. Raw Waveforms for sample inspection if requested
        raw_imu_df = None
        raw_semg_df = None
        if generate_raw_waveforms:
            raw_imu_df, q_imu = self.imu_sim.generate_raw_gait_waveforms(imu_feats)
            raw_semg_df, q_semg = self.semg_sim.generate_raw_semg_bursts(semg_feats)

        # Record for modeling (NO LATENT VARIABLES IN MODELING TABLE)
        patient_record = {
            'synthetic_patient_id': pid,
            'source_type': 'SYNTHETIC',
            'source_dataset': 'arthroscan_synthetic_multimodal_v1.0',
            'source_subject_id': pid,
            'synthetic_class': synthetic_class,
            'is_koa': synthetic_class,
            'class_label': synthetic_class,
            'age': sim_age,
            'gender': sim_gender,
            'bmi': sim_bmi,
            'kl_grade': sim_kl,
            
            # Missingness and Quality
            'has_imu': has_imu,
            'has_semg': has_semg,
            'imu_mask': has_imu,
            'semg_mask': has_semg,
            'imu_quality': q_imu,
            'semg_quality': q_semg,
            'split': 'TRAIN' # Synthetic data is exclusively for augmentation/training
        }
        
        # Prefix IMU features
        for k, v in imu_feats.items():
            patient_record[f"imu_{k}"] = v if has_imu else np.nan
            
        # Prefix sEMG features
        for k, v in semg_feats.items():
            patient_record[f"semg_{k}"] = v if has_semg else np.nan
            
        # Audit record (preserves latent variables for leakage check only)
        audit_record = {
            'synthetic_patient_id': pid,
            'synthetic_class': synthetic_class,
            'latent_functional_impairment': latent_func,
            'latent_mechanical_impairment': latent_mech,
            'latent_neuromuscular_compensation': latent_neuro
        }
        
        return patient_record, audit_record, raw_imu_df, raw_semg_df

    def generate_cohort(self, num_patients=5000, koa_ratio=0.50, raw_sample_count=5):
        """
        Generates an entire synthetic multimodal cohort.
        """
        patients = []
        audits = []
        sample_raws = {'imu': [], 'semg': []}
        
        for i in range(1, num_patients + 1):
            is_koa = (self.rng.rand() < koa_ratio)
            gen_raw = (i <= raw_sample_count)
            prec, arec, r_imu, r_semg = self.generate_single_patient(i, is_koa=is_koa, generate_raw_waveforms=gen_raw)
            patients.append(prec)
            audits.append(arec)
            if gen_raw:
                sample_raws['imu'].append((prec['synthetic_patient_id'], r_imu))
                sample_raws['semg'].append((prec['synthetic_patient_id'], r_semg))
                
        df_patients = pd.DataFrame(patients)
        df_audits = pd.DataFrame(audits)
        return df_patients, df_audits, sample_raws
