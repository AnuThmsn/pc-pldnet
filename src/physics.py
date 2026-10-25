"""
physics.py — Differentiable ASL forward model for PC-PLDNet.

Implements the Buxton-style pCASL kinetic model matching the repository's
data_gen.py: compute_signals_vec() exactly, but in PyTorch with full
gradient support.

Signal convention (from data_gen.py):
  - Clean signal S = prefix * e_att * (t1 - t2)
  - Noisy observation = dual-Rician magnitude sum:
      mc = sqrt((S + e1)^2 + e2^2)
      ml = sqrt((S + e3)^2 + e4^2)
      X  = mc + ml
  - The physics loss reconstructs the clean signal and compares against
    the de-noised expected value of X.

All tensor operations are differentiable.
"""

import torch
import torch.nn as nn
import numpy as np


class ASLForwardModel(nn.Module):
    """Differentiable Buxton-style pCASL forward model.

    Given CBF (ml/100g/min) and ATT (s), produces the expected clean
    ASL difference signal at each PLD, scaled by SCALE.

    This exactly mirrors data_gen.py: compute_signals_vec().
    """

    def __init__(self, plds, tau=1.8, T1t=1.2, T1a=1.66,
                 alpha=0.85, beta=0.75, lmbda=0.9, SCALE=100000.0):
        super().__init__()
        # Register as buffers so they move with .to(device)
        self.register_buffer('plds', torch.tensor(plds, dtype=torch.float32))
        self.tau = tau
        self.T1t = T1t
        self.T1a = T1a
        self.alpha = alpha
        self.beta = beta
        self.lmbda = lmbda
        self.SCALE = SCALE

    def forward(self, cbf, att):
        """
        Args:
            cbf: (B,) CBF in ml/100g/min
            att: (B,) ATT in seconds

        Returns:
            signals: (B, n_plds) clean ASL difference signal * SCALE
        """
        # Convert CBF to per-second flow
        f_per_s = cbf / (6000.0 * self.lmbda)   # (B,)
        f_per_s = f_per_s.unsqueeze(1)            # (B, 1)
        delta = att.unsqueeze(1)                  # (B, 1)
        pld = self.plds.unsqueeze(0)              # (1, n_plds)

        prefix = 2.0 * self.alpha * self.beta * self.T1t * (1.0 / self.lmbda) * f_per_s
        e_att = torch.exp(-delta / self.T1a)

        t1 = torch.exp(-torch.clamp(pld - delta, min=0.0) / self.T1t)
        t2 = torch.exp(-torch.clamp(self.tau + pld - delta, min=0.0) / self.T1t)

        signals = prefix * e_att * (t1 - t2) * self.SCALE
        return signals

    def expected_observation(self, cbf, att):
        """Returns the expected value of the dual-Rician magnitude sum.

        For the physics loss, we compare the predicted clean signal * 2
        (since X = mc + ml, and E[mc] ≈ E[ml] ≈ S for high SNR).
        In practice, E[sqrt((S+e)^2 + e^2)] ≈ S for SNR > 5, so
        E[X] ≈ 2*S is a reasonable approximation.

        Returns:
            expected_obs: (B, n_plds)
        """
        return 2.0 * self.forward(cbf, att)


def validate_forward_model(plds, tau, T1t, T1a, alpha, beta, lmbda, SCALE):
    """Validate PyTorch forward model against numpy reference."""
    model = ASLForwardModel(plds, tau, T1t, T1a, alpha, beta, lmbda, SCALE)
    model.eval()

    # Test case: known CBF and ATT
    cbf_np = np.array([50.0, 30.0, 70.0], dtype=np.float64)
    att_np = np.array([1.5, 2.0, 1.0], dtype=np.float64)

    # NumPy reference (from data_gen.py)
    plds_np = np.array(plds)
    f_per_s = (cbf_np / (6000.0 * lmbda))[:, None]
    delta = att_np[:, None]
    pld = plds_np[None, :]
    prefix = 2.0 * alpha * beta * T1t * (1.0 / lmbda) * f_per_s
    e_att = np.exp(-delta / T1a)
    t1 = np.exp(-np.maximum(pld - delta, 0.0) / T1t)
    t2 = np.exp(-np.maximum(tau + pld - delta, 0.0) / T1t)
    ref_signals = prefix * e_att * (t1 - t2) * SCALE

    # PyTorch
    cbf_t = torch.tensor(cbf_np, dtype=torch.float32)
    att_t = torch.tensor(att_np, dtype=torch.float32)
    with torch.no_grad():
        pt_signals = model(cbf_t, att_t).numpy()

    max_err = np.max(np.abs(ref_signals - pt_signals))
    rel_err = max_err / (np.max(np.abs(ref_signals)) + 1e-10)
    return max_err, rel_err, ref_signals, pt_signals

