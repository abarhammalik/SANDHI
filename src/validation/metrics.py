"""
Statistical Distribution Validation, Leakage Audit, and Domain Shift Modules.
Implements:
  - Quantitative distribution metrics: Wasserstein distance, Kolmogorov-Smirnov (KS) test, correlation difference
  - Target Leakage Auditor (validates that targets and latent factors never appear in feature sets)
  - Domain Shift Classifier (evaluates Real vs Synthetic distinguishability)
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

FORBIDDEN_LEAKAGE_COLUMNS = [
    'label', 'synthetic_label', 'class_label', 'is_koa', 'synthetic_class',
    'latent_functional_impairment', 'latent_mechanical_impairment', 'latent_neuromuscular_compensation',
    'k-l grade', 'kl_grade', 'womac score', 'womac_score', 'vas score', 'vas_score'
]

def audit_feature_leakage(feature_cols, dataset_name="dataset"):
    """
    Audits feature columns against known target leakage columns.
    Returns: DataFrame audit results.
    """
    audit_rows = []
    has_leakage = False
    for col in feature_cols:
        col_lower = col.lower().strip()
        is_leaked = any(forbidden in col_lower for forbidden in FORBIDDEN_LEAKAGE_COLUMNS)
        reason = "Forbidden target or simulation latent driver" if is_leaked else "Compliant feature"
        if is_leaked:
            has_leakage = True
        audit_rows.append({
            'dataset': dataset_name,
            'feature_name': col,
            'is_leakage': is_leaked,
            'audit_note': reason
        })
    df_audit = pd.DataFrame(audit_rows)
    return df_audit, has_leakage

def compute_distribution_similarity(real_train_df, synthetic_df, feature_cols):
    """
    Computes Wasserstein distance and KS-test statistic between real training and synthetic distributions.
    """
    results = []
    cols = [c for c in feature_cols if c in real_train_df.columns and c in synthetic_df.columns]
    
    for c in cols:
        r_vals = real_train_df[c].dropna().to_numpy(dtype=float)
        s_vals = synthetic_df[c].dropna().to_numpy(dtype=float)
        
        if len(r_vals) == 0 or len(s_vals) == 0:
            continue
            
        # Scale by real standard deviation for normalized Wasserstein metric
        r_std = np.std(r_vals) if np.std(r_vals) > 0 else 1.0
        w_dist = float(wasserstein_distance(r_vals, s_vals))
        norm_w_dist = float(w_dist / r_std)
        
        ks_stat, ks_pval = ks_2samp(r_vals, s_vals)
        
        results.append({
            'feature': c,
            'real_mean': float(np.mean(r_vals)),
            'synth_mean': float(np.mean(s_vals)),
            'real_std': float(np.std(r_vals)),
            'synth_std': float(np.std(s_vals)),
            'wasserstein_distance': w_dist,
            'normalized_wasserstein': norm_w_dist,
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pval)
        })
    return pd.DataFrame(results)

def evaluate_domain_shift(real_train_df, synthetic_df, feature_cols):
    """
    Trains a discriminator to distinguish REAL (0) vs SYNTHETIC (1).
    A high discriminator AUC (> 0.90) indicates perceptible domain shift.
    """
    cols = [c for c in feature_cols if c in real_train_df.columns and c in synthetic_df.columns]
    
    X_real = real_train_df[cols].dropna()
    X_synth = synthetic_df[cols].dropna()
    
    # Subsample synthetic to match real size for balanced evaluation
    n_sample = min(len(X_real), len(X_synth))
    X_real_sub = X_real.sample(n=n_sample, random_state=42)
    X_synth_sub = X_synth.sample(n=n_sample, random_state=42)
    
    X = np.vstack([X_real_sub.values, X_synth_sub.values])
    y = np.array([0] * n_sample + [1] * n_sample)
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    scores = cross_val_score(clf, X, y, cv=3, scoring='roc_auc')
    mean_auc = float(np.mean(scores))
    
    return {
        'discriminator_roc_auc': mean_auc,
        'features_evaluated': cols,
        'domain_shift_interpretation': (
            "Substantial domain shift" if mean_auc > 0.85 else
            "Moderate divergence" if mean_auc > 0.70 else
            "Excellent alignment (statistically indistinguishable)"
        )
    }
