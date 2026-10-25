"""
metrics.py — Evaluation metrics for CBF and ATT estimation.
"""

import numpy as np


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mean_bias(y_true, y_pred):
    return np.mean(y_pred - y_true)


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-10)


def pearson_r(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def concordance_correlation_coefficient(y_true, y_pred):
    """Lin's concordance correlation coefficient (CCC)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covar = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom < 1e-10:
        return 0.0
    return float(2.0 * covar / denom)


def compute_all_metrics(y_true, y_pred, name=""):
    """Compute all metrics for a single parameter.

    Args:
        y_true: (N,) ground truth
        y_pred: (N,) predictions
        name: parameter name (CBF or ATT)

    Returns:
        dict of metric name → value
    """
    prefix = f"{name}_" if name else ""
    return {
        f'{prefix}RMSE': float(rmse(y_true, y_pred)),
        f'{prefix}MAE': float(mae(y_true, y_pred)),
        f'{prefix}Bias': float(mean_bias(y_true, y_pred)),
        f'{prefix}R2': float(r_squared(y_true, y_pred)),
        f'{prefix}Pearson_r': float(pearson_r(y_true, y_pred)),
        f'{prefix}CCC': float(concordance_correlation_coefficient(y_true, y_pred)),
    }


def compute_snr_stratified_metrics(y_true_cbf, y_pred_cbf,
                                    y_true_att, y_pred_att,
                                    snr_values, snr_bins):
    """Compute metrics stratified by SNR.

    Args:
        snr_bins: list of (low, high) tuples defining SNR ranges

    Returns:
        list of dicts, one per SNR bin
    """
    results = []
    for snr_low, snr_high in snr_bins:
        mask = (snr_values >= snr_low) & (snr_values < snr_high)
        n = mask.sum()
        if n == 0:
            continue

        label = f"SNR_{snr_low:.0f}" if not np.isinf(snr_high) else f"SNR_{snr_low:.0f}+"
        row = {
            'SNR_range': f"[{snr_low:.0f}, {snr_high:.0f})" if not np.isinf(snr_high) else f"[{snr_low:.0f}, ∞)",
            'N_samples': int(n),
        }
        row.update(compute_all_metrics(y_true_cbf[mask], y_pred_cbf[mask], 'CBF'))
        row.update(compute_all_metrics(y_true_att[mask], y_pred_att[mask], 'ATT'))
        results.append(row)
    return results

