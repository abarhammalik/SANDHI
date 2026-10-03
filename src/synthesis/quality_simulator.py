"""
Quality Degradation and Artifact Simulation Module.
Simulates realistic real-world sensor artifacts:
  - IMU: sensor displacement, baseline drift, sample dropouts, motion shocks
  - sEMG: electrode contact impedance noise, baseline wander, power-line hum, SNR reduction
  - Computes continuous modality quality scores (0.0 to 1.0) and missingness masks
"""
import numpy as np

class QualitySimulator:
    def __init__(self, random_state=42):
        self.rng = np.random.RandomState(random_state)

    def degrade_imu_signal(self, acc_mag, gyr_mag, fs=100.0):
        """
        Applies stochastic degradation to IMU signals and returns degraded signals + quality score.
        """
        degraded_acc = acc_mag.copy()
        degraded_gyr = gyr_mag.copy()
        
        quality = 1.0
        
        # 1. Baseline Drift (low-frequency polynomial or random walk)
        if self.rng.rand() < 0.15:
            drift_amplitude = self.rng.uniform(0.1, 0.5)
            t = np.linspace(0, 1, len(degraded_acc))
            drift = drift_amplitude * np.sin(2 * np.pi * 0.1 * t)
            degraded_acc += drift
            quality -= 0.10
            
        # 2. Motion Shocks / Artifact Spikes (abrupt sensor displacement)
        if self.rng.rand() < 0.12:
            num_spikes = self.rng.randint(1, 4)
            spike_locs = self.rng.randint(0, len(degraded_acc), size=num_spikes)
            for loc in spike_locs:
                spike_width = self.rng.randint(2, 6)
                end_loc = min(loc + spike_width, len(degraded_acc))
                degraded_acc[loc:end_loc] += self.rng.normal(2.0, 0.5)
                degraded_gyr[loc:end_loc] += self.rng.normal(1.5, 0.4)
            quality -= 0.15
            
        # 3. Missing Samples (packet loss dropouts)
        if self.rng.rand() < 0.10:
            dropout_rate = self.rng.uniform(0.01, 0.05)
            mask = self.rng.rand(len(degraded_acc)) < dropout_rate
            degraded_acc[mask] = 0.0
            degraded_gyr[mask] = 0.0
            quality -= 0.15
            
        # Add high-frequency thermal noise
        noise_level = self.rng.uniform(0.01, 0.08)
        degraded_acc += self.rng.normal(0, noise_level, size=len(degraded_acc))
        degraded_gyr += self.rng.normal(0, noise_level, size=len(degraded_gyr))
        quality -= noise_level * 0.5
        
        quality_score = float(np.clip(quality, 0.20, 1.0))
        return degraded_acc, degraded_gyr, quality_score

    def degrade_semg_signal(self, raw_signal, fs=2000.0):
        """
        Applies electrode contact impedance artifacts, 50 Hz hum, and baseline wander to sEMG.
        """
        degraded = raw_signal.copy()
        n_samples = len(degraded)
        t = np.arange(n_samples) / fs
        quality = 1.0
        
        # 1. Baseline Wander (respiration / perspiration movement < 1 Hz)
        if self.rng.rand() < 0.18:
            wander_freq = self.rng.uniform(0.2, 0.8)
            wander_amp = self.rng.uniform(0.02, 0.10) * np.max(np.abs(degraded) + 1e-4)
            degraded += wander_amp * np.sin(2 * np.pi * wander_freq * t)
            quality -= 0.12
            
        # 2. Power-line Interference (50 Hz hum from poor grounding)
        if self.rng.rand() < 0.15:
            hum_amp = self.rng.uniform(0.01, 0.05) * np.max(np.abs(degraded) + 1e-4)
            degraded += hum_amp * np.sin(2 * np.pi * 50.0 * t + self.rng.uniform(0, 2*np.pi))
            quality -= 0.10
            
        # 3. Electrode Contact Impedance Noise / Clipping
        if self.rng.rand() < 0.10:
            contact_noise = self.rng.normal(0, 0.05 * np.std(degraded + 1e-4), size=n_samples)
            degraded += contact_noise
            quality -= 0.15
            
        quality_score = float(np.clip(quality, 0.20, 1.0))
        return degraded, quality_score

    def generate_missingness_masks(self, p_missing_imu=0.08, p_missing_semg=0.08):
        """Generates modality availability flags for missing-modality testing."""
        has_imu = 1 if self.rng.rand() >= p_missing_imu else 0
        has_semg = 1 if self.rng.rand() >= p_missing_semg else 0
        # Ensure at least one modality is present
        if has_imu == 0 and has_semg == 0:
            if self.rng.rand() < 0.5:
                has_imu = 1
            else:
                has_semg = 1
        return has_imu, has_semg
