# ARTHROSCAN-NER Data Inventory & Preprocessing Justification Report

**Date:** 2026-09-30
**Status:** Real Datasets Fully Verified and Inspected

## 1. Executive Summary

The ARTHROSCAN-NER pipeline is grounded in **two authentic clinical datasets** acquired locally:

1. **Clinical Gait Inertial Signals (IMU)**: 260 subjects, 1,356 trials, 4 wearable IMU sensors (Head, Lower Back, Left Foot, Right Foot) recording 10 channels at 100 Hz. Contains 73 Healthy Subjects (HS), 18 Knee Osteoarthritis (KOA) subjects, 15 Hip Osteoarthritis (HOA), 11 ACL injuries, and 143 neurological subjects.
2. **Quadriceps Surface Electromyography (sEMG)**: 88 subjects (43 Knee Osteoarthritis patients, 45 age-matched Healthy Controls) across 3 distinct tasks (Treadmill Walking, Up-Down Stairs, Maximum Voluntary Contraction / MVC) sampled at 2000 Hz. Focuses directly on quadriceps muscle balance: Vastus Medialis (VM) and Vastus Lateralis (VL), plus RF and BF, alongside comprehensive clinical ground truth (Kellgren-Lawrence grade 1-4, WOMAC, VAS, BMI, JPR).

## 2. Dataset Inventory Table

### Clinical Gait Inertial Signals (Figshare 28806086) (IMU)

- **Modality:** IMU
- **Local Path:** `data/real/imu/dataset/data`
- **Subject Count:** 260
- **Trial Count:** 1356
- **File Formats:** .txt (tab-delimited raw & processed), .json (meta), .png (plot)
- **Sampling Frequency:** 100 Hz
- **Sensor Placements:** Head (HE), Lower Back (LB), Left Foot (LF), Right Foot (RF)
- **Channels Available:** Acc_X, Acc_Y, Acc_Z, Gyr_X, Gyr_Y, Gyr_Z, Mag_X, Mag_Y, Mag_Z (per sensor = 40 channels)
- **Cohorts & Classes:** healthy (HS: 73), ortho (KOA: 18, HOA: 15, ACL: 11), neuro (CIPN: 19, CVA: 49, PD: 24, RIL: 51)
- **Clinical Metadata:** Age, Gender, Height, Weight, BMI, Laterality, Deficit Side, Evaluation Scores, Gait Events (toe-off, heel-strike), U-turn boundaries
- **Estimated Duration:** 20.8 s (range 14.5 - 25.2 s)
- **Total Files & Size:** 9492 files, 4786.3 MB

### Quadriceps sEMG in Knee Osteoarthritis (Dryad fn2z34v8s) (sEMG)

- **Modality:** sEMG
- **Local Path:** `data/real/semg/raw/full sEMG data in correct format`
- **Subject Count:** 88
- **Trial Count:** 264
- **File Formats:** .csv (32 per task), .xlsx (56 per task), .csv summary (full_indicators_summary)
- **Sampling Frequency:** 2000 Hz
- **Sensor Placements:** Bilateral Vastus Medialis (VM, L/R), Vastus Lateralis (VL, L/R), Rectus Femoris (RF, L/R), Biceps Femoris (BF, L/R)
- **Channels Available:** Raw sEMG CH1/CH2 per sensor: VM (IMU 1 & 9), RF (IMU 2 & 10), VL (IMU 3 & 11), BF (IMU 4/5 & 12/13). Key channel: CH2
- **Cohorts & Classes:** KOA (Knee Osteoarthritis): 43, Healthy Controls: 45
- **Clinical Metadata:** Age, Gender, BMI, K-L grade (1-4), KOA period, VAS score, WOMAC score, 30/45/60 deg Joint Position Reproduction (JPR)
- **Estimated Duration:** ~15-20 s walking (15 gait cycles) at 2000 Hz (~30k-40k samples/channel)
- **Total Files & Size:** 267 files, 2169.6 MB

## 3. Critical Architectural Constraint: Cohort Independence

> **RULE: ABSOLUTELY DO NOT MIX UNRELATED PATIENTS ACROSS REAL COHORTS.**

- The IMU dataset (Figshare 28806086) and sEMG dataset (Dryad fn2z34v8s) represent **independent clinical patient populations**.
- It is strictly invalid to bind `IMU_HS_1` with `sEMG_subject_001` as a single real patient.
- Instead, real datasets are strictly used to:
  1. Train and evaluate unimodal real baseline models on untouched test sets.
  2. Characterize class-conditional joint distributions, covariance, and physiological parameters strictly from **training splits**.
  3. Calibrate the **Synthetic Multimodal Generator** to generate authentic paired `(synthetic_IMU, synthetic_sEMG)` samples sharing common latent biomechanical impairment parameters.

## 4. Signal Processing & Preprocessing Justifications

### A. IMU Signal Processing Pipeline
1. **Coordinate & Unit Standardisation:** Ensure 3-axis acceleration is in $m/s^2$ (sensor frame standard) and gyroscope angular velocity in $rad/s$.
2. **Gravity & Drift Correction:** Process lower back and foot trajectories using high-pass filtering (0.25 Hz cutoff) and complementary filtering to isolate dynamic linear acceleration from gravitational static acceleration.
3. **Gait Event Segmentation:** Leverage detected toe-off and heel-strike timestamps provided in official trial metadata (`leftGaitEvents`, `rightGaitEvents`) and foot angular velocity zero-crossings to extract stride time, stance time, swing time, cadence, and bilateral asymmetry.
4. **Spatiotemporal & Kinematic Features:** Extract mean, std, RMS, peak acceleration/velocity, stride time variability, and stance phase asymmetry.

### B. sEMG Signal Processing Pipeline
1. **Dual Format Ingestion:** The 88 subjects are split across `.csv` (32 subjects) and `.xlsx` (56 subjects). The pipeline standardizes channel mappings for both formats, extracting Channel 2 (`EMG_CH2`) which represents the calibrated bipolar differential recording for VM, VL, RF, and BF.
2. **Filtering (Bandpass + Notch):** Apply a 4th-order zero-phase Butterworth bandpass filter (20 - 450 Hz) matching standard SENIAM / clinical electromyography guidelines, plus a 50 Hz notch filter (Q=30) to eliminate European/Asian power-line interference.
3. **Rectification & Linear Envelope:** Full-wave rectification followed by low-pass filtering (6 Hz Butterworth) to construct the muscle activation linear envelope.
4. **Spectral Analysis (MDF):** Estimate the power spectral density (PSD) using Welch's periodogram (Hanning window, 500 ms length, 50% overlap) to compute the **Median Frequency (MDF)** ($f_{med}$ where $\int_0^{f_{med}} P(f)df = \frac{1}{2}\int_0^{\infty} P(f)df$).
5. **Muscle Balance Metrics:** Compute $VM / VL$ RMS ratio, bilateral asymmetry index, and MVC normalization using the dedicated MVC squat trials.

## 5. Provenance & Target Leakage Safeguards

- Every record across all feature tables includes `source_type` (`REAL`, `DERIVED_FROM_REAL`, `SYNTHETIC`, `DERIVED_FROM_SYNTHETIC`).
- Identifiers strictly follow: `real_imu_<id>`, `real_semg_<id>`, and `synthetic_patient_<06d>`.
- All latent simulation factors (`latent_functional_impairment`, `latent_mechanical_impairment`) are stripped prior to feature ingestion and never used as model inputs.
- Real test sets are held out at the participant level (70% train, 15% validation, 15% test) before any synthetic distribution calibration.
