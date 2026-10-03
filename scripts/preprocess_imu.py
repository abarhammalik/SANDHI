"""
IMU Preprocessing Pipeline Execution Script.
Wraps standardized ingestion, filtering, gravity compensation, and feature calculation.
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.extract_imu_features import run_imu_feature_extraction

def main():
    print("=== STARTING REAL IMU PREPROCESSING & FEATURE EXTRACTION ===")
    df = run_imu_feature_extraction()
    print("=== REAL IMU PREPROCESSING COMPLETED ===")

if __name__ == '__main__':
    main()
