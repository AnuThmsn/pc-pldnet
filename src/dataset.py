"""
dataset.py — Synthetic ASL data generation for PC-PLDNet.

Reuses the exact signal model, noise convention, and parameter ranges
from the repository's data_gen.py. Independent RNG seeds ensure
leakage-safe train/val/test splits.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── Physics constants (must match data_gen.py exactly) ──
PLDs = np.array([1.525, 2.025, 2.525, 3.025])
TAU = 1.8
T1T = 1.2
T1A = 1.66
ALPHA = 0.85
BETA = 0.75
LMBDA = 0.9
SCALE = 100_000.0
SIG_REF_SC = 334.2039
SD_MAX = SIG_REF_SC / 5.0

CBF_MIN, CBF_MAX = 0.0, 100.0
ATT_MIN, ATT_MAX = 0.5, 3.0


def compute_signals_vec(cbf, att, ld=TAU):
    """Exact copy of data_gen.py's signal model."""
    f_per_s = (cbf / (6000.0 * LMBDA))[:, None]
    delta = att[:, None]
    pld = PLDs[None, :]
    prefix = 2.0 * ALPHA * BETA * T1T * (1.0 / LMBDA) * f_per_s
    e_att = np.exp(-delta / T1A)
    t1 = np.exp(-np.maximum(pld - delta, 0.0) / T1T)
    t2 = np.exp(-np.maximum(ld + pld - delta, 0.0) / T1T)
    return prefix * e_att * (t1 - t2)


def generate_data(N_total, rng, n_noise_levels=100):
    """Generate synthetic ASL data with dual-Rician noise.

    Exact reproduction of data_gen.py: generate_data().

    Returns:
        X: (N, 4) noisy observations (mc + ml), float32
        Y: (N, 2) ground truth [CBF, ATT], float32
        sd_arr: (N,) noise SD for each sample, float64
    """
    cbf_gt = rng.uniform(CBF_MIN, CBF_MAX, N_total).astype(np.float64)
    att_gt = rng.uniform(ATT_MIN, ATT_MAX, N_total).astype(np.float64)

    sd_levels = np.linspace(0.0, SD_MAX, n_noise_levels)
    sd_choice_idx = rng.integers(0, n_noise_levels, size=N_total)
    sd_arr = sd_levels[sd_choice_idx]

    sig = compute_signals_vec(cbf_gt, att_gt) * SCALE

    sd = sd_arr[:, None]
    e1 = rng.normal(0.0, 1.0, size=sig.shape) * sd
    e2 = rng.normal(0.0, 1.0, size=sig.shape) * sd
    e3 = rng.normal(0.0, 1.0, size=sig.shape) * sd
    e4 = rng.normal(0.0, 1.0, size=sig.shape) * sd

    mc = np.sqrt((sig + e1) ** 2 + e2 ** 2)
    ml = np.sqrt((sig + e3) ** 2 + e4 ** 2)
    X = (mc + ml).astype(np.float32)

    Y = np.stack([cbf_gt, att_gt], axis=1).astype(np.float32)
    return X, Y, sd_arr


def compute_snr_per_sample(X_clean_scaled, sd_arr):
    """Compute per-sample SNR = RMS(clean_signal) / sd."""
    rms = np.sqrt(np.mean(X_clean_scaled ** 2, axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(sd_arr > 0, rms / sd_arr, np.inf)
    return snr


class ASLDataset(Dataset):
    """PyTorch Dataset for ASL voxel-wise data."""

    def __init__(self, X, Y, X_mean=None, X_std=None):
        """
        Args:
            X: (N, 4) raw observations
            Y: (N, 2) [CBF, ATT] ground truth
            X_mean, X_std: normalization stats (from training set only)
        """
        if X_mean is not None and X_std is not None:
            X = (X - X_mean) / X_std
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def create_datasets(cfg):
    """Create train/val/test datasets and normalization stats.

    Args:
        cfg: config dict with dataset section

    Returns:
        dict with datasets, loaders, normalization stats, and metadata
    """
    ds_cfg = cfg['dataset']
    N_TRAIN = ds_cfg['N_TRAIN']
    N_VAL = ds_cfg['N_VAL']
    N_TEST = ds_cfg['N_TEST']

    rng_tr = np.random.default_rng(ds_cfg['seed_train'])
    rng_val = np.random.default_rng(ds_cfg['seed_val'])
    rng_test = np.random.default_rng(ds_cfg['seed_test'])

    print(f"Generating training data ({N_TRAIN} samples)...")
    X_tr, Y_tr, sd_tr = generate_data(N_TRAIN, rng_tr, ds_cfg['N_NOISE_LEVELS_TRAIN'])

    print(f"Generating validation data ({N_VAL} samples)...")
    X_val, Y_val, sd_val = generate_data(N_VAL, rng_val, ds_cfg['N_NOISE_LEVELS_VAL'])

    print(f"Generating test data ({N_TEST} samples)...")
    X_test, Y_test, sd_test = generate_data(N_TEST, rng_test, ds_cfg['N_NOISE_LEVELS_TEST'])

    # Normalization stats from training set only
    X_mean = X_tr.mean(axis=0, keepdims=True).astype(np.float32)
    X_std = X_tr.std(axis=0, keepdims=True).astype(np.float32) + 1e-8

    # Also compute clean signals for SNR computation in test
    cbf_test = Y_test[:, 0].astype(np.float64)
    att_test = Y_test[:, 1].astype(np.float64)
    clean_signals_test = compute_signals_vec(cbf_test, att_test) * SCALE

    snr_test = compute_snr_per_sample(clean_signals_test, sd_test)

    # Create datasets
    train_ds = ASLDataset(X_tr, Y_tr, X_mean, X_std)
    val_ds = ASLDataset(X_val, Y_val, X_mean, X_std)
    test_ds = ASLDataset(X_test, Y_test, X_mean, X_std)

    bs = cfg['training']['batch_size']

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs * 2, shuffle=False,
                             num_workers=0, pin_memory=True)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'X_mean': X_mean,
        'X_std': X_std,
        'X_test_raw': X_test,
        'Y_test': Y_test,
        'sd_test': sd_test,
        'snr_test': snr_test,
        'clean_signals_test': clean_signals_test,
        'N_TRAIN': N_TRAIN,
        'N_VAL': N_VAL,
        'N_TEST': N_TEST,
    }

