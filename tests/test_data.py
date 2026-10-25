"""
test_data.py — Automated tests for dataset generation, splits, and metrics.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import generate_data, compute_signals_vec, CBF_MIN, CBF_MAX, ATT_MIN, ATT_MAX
from src.metrics import (rmse, mae, mean_bias, r_squared,
                          concordance_correlation_coefficient, compute_all_metrics)


def test_data_generation_shapes_and_ranges():
    n_samples = 500
    rng = np.random.default_rng(123)
    X, Y, sd_arr = generate_data(n_samples, rng, n_noise_levels=10)

    assert X.shape == (n_samples, 4), f"X shape {X.shape} != ({n_samples}, 4)"
    assert Y.shape == (n_samples, 2), f"Y shape {Y.shape} != ({n_samples}, 2)"
    assert sd_arr.shape == (n_samples,), f"sd_arr shape {sd_arr.shape} != ({n_samples},)"

    assert np.isfinite(X).all(), "X contains non-finite values"
    assert np.isfinite(Y).all(), "Y contains non-finite values"
    assert (X >= 0.0).all(), "Dual-Rician magnitude sum must be non-negative"

    # Targets in valid range
    cbf = Y[:, 0]
    att = Y[:, 1]
    assert (cbf >= CBF_MIN).all() and (cbf <= CBF_MAX).all()
    assert (att >= ATT_MIN).all() and (att <= ATT_MAX).all()


def test_training_only_normalization():
    rng_tr = np.random.default_rng(42)
    rng_val = np.random.default_rng(99)

    X_tr, _, _ = generate_data(1000, rng_tr)
    X_val, _, _ = generate_data(200, rng_val)

    X_mean = X_tr.mean(axis=0, keepdims=True)
    X_std = X_tr.std(axis=0, keepdims=True) + 1e-8

    X_tr_norm = (X_tr - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std

    # Train normalized should have zero mean and unit std
    assert np.allclose(X_tr_norm.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(X_tr_norm.std(axis=0), 1.0, atol=1e-3)

    # Val normalized uses train stats, so its mean/std will not be exactly 0/1
    val_mean = X_val_norm.mean(axis=0)
    assert not np.allclose(val_mean, 0.0, atol=1e-7), "Validation appears to use its own mean"


def test_metric_calculations_verifiable():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    # 1. Perfect prediction
    y_perfect = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    m_perfect = compute_all_metrics(y_true, y_perfect)
    assert np.isclose(m_perfect['RMSE'], 0.0)
    assert np.isclose(m_perfect['MAE'], 0.0)
    assert np.isclose(m_perfect['Bias'], 0.0)
    assert np.isclose(m_perfect['R2'], 1.0)
    assert np.isclose(m_perfect['CCC'], 1.0)

    # 2. Constant positive bias (+2.0)
    y_biased = y_true + 2.0
    m_biased = compute_all_metrics(y_true, y_biased)
    assert np.isclose(m_biased['RMSE'], 2.0)
    assert np.isclose(m_biased['MAE'], 2.0)
    assert np.isclose(m_biased['Bias'], 2.0)
    assert np.isclose(m_biased['R2'], 1.0 - 4.0 / np.var(y_true))
    assert m_biased['Pearson_r'] > 0.9999  # Correlation is still 1.0

    # 3. Known error: errors = [-1, 2, -3, 4, -5]
    y_err = y_true + np.array([-1.0, 2.0, -3.0, 4.0, -5.0])
    m_err = compute_all_metrics(y_true, y_err)
    expected_mae = (1.0 + 2.0 + 3.0 + 4.0 + 5.0) / 5.0  # 3.0
    expected_rmse = np.sqrt((1 + 4 + 9 + 16 + 25) / 5.0)  # sqrt(11) = 3.31662479
    expected_bias = (-1.0 + 2.0 - 3.0 + 4.0 - 5.0) / 5.0  # -0.6
    assert np.isclose(m_err['MAE'], expected_mae)
    assert np.isclose(m_err['RMSE'], expected_rmse)
    assert np.isclose(m_err['Bias'], expected_bias)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

