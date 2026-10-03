"""
sEMG Preprocessing Pipeline Execution Script.
Wraps ingestion, bandpass/notch filtering, rectification, envelope extraction, and feature extraction.
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.extract_semg_features import run_semg_feature_extraction

def main():
    print("=== STARTING REAL sEMG PREPROCESSING & FEATURE EXTRACTION ===")
    df = run_semg_feature_extraction()
    print("=== REAL sEMG PREPROCESSING COMPLETED ===")

if __name__ == '__main__':
    main()
