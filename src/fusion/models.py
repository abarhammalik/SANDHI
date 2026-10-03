"""
Multimodal Fusion Architecture Module.
Supports:
  1. Early Feature Fusion (concatenated features + modality masks + quality weighting)
  2. Late Probability Fusion (calibrated unimodal probabilities -> meta-classifier)
Classifiers:
  - Logistic Regression (interpretable linear baseline)
  - Random Forest
  - XGBoost
Handles modality masking (IMU only, sEMG only, IMU + sEMG) and quality scores.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
)

def evaluate_predictions(y_true, y_pred, y_prob=None):
    """Computes comprehensive binary classification metrics."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0))
    }
    
    # Specificity and Sensitivity from Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics['confusion_matrix'] = cm.tolist()
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics['roc_auc'] = float(np.nan)
        try:
            metrics['pr_auc'] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics['pr_auc'] = float(np.nan)
        try:
            metrics['brier_score'] = float(brier_score_loss(y_true, y_prob))
        except Exception:
            metrics['brier_score'] = float(np.nan)
    else:
        metrics['roc_auc'] = float(np.nan)
        metrics['pr_auc'] = float(np.nan)
        metrics['brier_score'] = float(np.nan)
        
    return metrics

class EarlyFusionModel:
    """Concatenation of IMU + sEMG features with modality masks and quality scores."""
    def __init__(self, model_type='xgboost', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=random_state)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)
        elif model_type == 'xgboost':
            self.model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                      random_state=random_state, eval_metric='logloss')
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class LateProbabilityFusionModel:
    """Meta-classifier fusing unimodal prediction probabilities and quality scores."""
    def __init__(self, meta_model_type='logistic_regression', random_state=42):
        self.meta_model_type = meta_model_type
        self.random_state = random_state
        if meta_model_type == 'logistic_regression':
            self.meta_model = LogisticRegression(max_iter=1000, random_state=random_state)
        elif meta_model_type == 'random_forest':
            self.meta_model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=random_state)
        elif meta_model_type == 'xgboost':
            self.meta_model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05,
                                           random_state=random_state, eval_metric='logloss')

    def fit(self, imu_prob, semg_prob, imu_mask, semg_mask, imu_q, semg_q, y):
        """Fits meta-model on unimodal probabilities and confidence flags."""
        features = np.column_stack([imu_prob, semg_prob, imu_mask, semg_mask, imu_q, semg_q])
        self.meta_model.fit(features, y)
        return self

    def predict_proba(self, imu_prob, semg_prob, imu_mask, semg_mask, imu_q, semg_q):
        features = np.column_stack([imu_prob, semg_prob, imu_mask, semg_mask, imu_q, semg_q])
        return self.meta_model.predict_proba(features)

    def predict(self, imu_prob, semg_prob, imu_mask, semg_mask, imu_q, semg_q):
        features = np.column_stack([imu_prob, semg_prob, imu_mask, semg_mask, imu_q, semg_q])
        return self.meta_model.predict(features)
