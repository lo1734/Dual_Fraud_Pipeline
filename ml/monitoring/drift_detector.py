import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import wasserstein_distance, ks_2samp

class FeatureDriftDetector:
    """
    Monitors distribution shifts across incoming features and model predictions
    using Population Stability Index (PSI) and Wasserstein Distance.
    """
    def __init__(self, baseline_df: pd.DataFrame, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.baseline_stats = {}
        self._fit_baseline(baseline_df)

    def _fit_baseline(self, df: pd.DataFrame):
        for col in self.feature_cols:
            vals = df[col].dropna().values.astype(float)
            unique_vals = np.unique(vals)
            if len(unique_vals) <= 10:
                # Discrete count feature (e.g. 0, 1, 2, ...)
                # Create discrete bins centered on integers
                bins = np.concatenate([[-np.inf], unique_vals[:-1] + 0.5, [np.inf]])
            else:
                # Continuous feature (e.g. amount, seconds)
                quantiles = np.linspace(0, 1, 11)
                bins = np.unique(np.quantile(vals, quantiles))
                bins[0] = -np.inf
                bins[-1] = np.inf
                if len(bins) < 3:
                    bins = np.array([-np.inf, np.median(vals), np.inf])

            expected_counts, _ = np.histogram(vals, bins=bins)
            expected_pct = (expected_counts + 1e-3) / (len(vals) + 1e-3 * len(expected_counts))

            self.baseline_stats[col] = {
                "bins": bins,
                "expected_pct": expected_pct,
                "values": vals
            }

    def compute_feature_psi(self, current_df: pd.DataFrame, col: str) -> float:
        """Computes Population Stability Index (PSI) for a specific feature."""
        if col not in self.baseline_stats:
            return 0.0

        current_vals = current_df[col].dropna().values.astype(float)
        if len(current_vals) == 0:
            return 0.0

        bins = self.baseline_stats[col]["bins"]
        expected_pct = self.baseline_stats[col]["expected_pct"]

        actual_counts, _ = np.histogram(current_vals, bins=bins)
        actual_pct = (actual_counts + 1e-4) / (len(current_vals) + 1e-4 * len(actual_counts))

        # PSI formula: sum((actual - expected) * ln(actual / expected))
        psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi_val)

    def compute_all_psi(self, current_df: pd.DataFrame) -> Dict[str, float]:
        """Computes PSI across all monitored features."""
        results = {}
        for col in self.feature_cols:
            if col in current_df.columns:
                results[col] = round(self.compute_feature_psi(current_df, col), 4)
        return results

    def compute_wasserstein(self, current_df: pd.DataFrame, col: str) -> float:
        """Computes Wasserstein earth mover's distance for continuous distributions."""
        if col not in self.baseline_stats:
            return 0.0
        base_vals = self.baseline_stats[col]["values"]
        curr_vals = current_df[col].dropna().values.astype(float)
        return float(wasserstein_distance(base_vals, curr_vals))

    def evaluate_drift_report(self, current_df: pd.DataFrame) -> dict:
        """
        Generates comprehensive drift assessment:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.25: Moderate change / alert
        - PSI >= 0.25: Significant drift / retrain trigger
        """
        psi_dict = self.compute_all_psi(current_df)
        drift_alerts = []
        retrain_triggers = []

        for col, psi in psi_dict.items():
            if psi >= 0.25:
                retrain_triggers.append(f"{col} (PSI={psi})")
            elif psi >= 0.10:
                drift_alerts.append(f"{col} (PSI={psi})")

        return {
            "feature_psi": psi_dict,
            "moderate_drift_features": drift_alerts,
            "significant_drift_features": retrain_triggers,
            "action_required": "RETRAIN" if len(retrain_triggers) > 0 else ("INVESTIGATE" if len(drift_alerts) > 0 else "NONE")
        }

def compute_model_score_divergence(fast_scores: np.ndarray, deep_scores: np.ndarray) -> dict:
    """
    Computes divergence between fast-model predictions and deep-model predictions
    to detect if fast path is drifting away from deep path quality.
    """
    abs_diff = np.abs(fast_scores - deep_scores)
    mean_abs_error = float(np.mean(abs_diff))
    p95_diff = float(np.percentile(abs_diff, 95.0))
    correlation = float(np.corrcoef(fast_scores, deep_scores)[0, 1])

    return {
        "mean_absolute_divergence": round(mean_abs_error, 4),
        "p95_divergence": round(p95_diff, 4),
        "score_correlation": round(correlation, 4),
        "divergence_alert": bool(mean_abs_error > 0.08 or correlation < 0.70)
    }
