"""
test_physics.py — Automated tests for ASL forward model.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.physics import ASLForwardModel, validate_forward_model


def test_forward_model_against_numpy_reference():
    plds = [1.525, 2.025, 2.525, 3.025]
    tau = 1.8
    T1t = 1.2
    T1a = 1.66
    alpha = 0.85
    beta = 0.75
    lmbda = 0.9
    scale = 100000.0

    max_err, rel_err, ref_sig, pt_sig = validate_forward_model(
        plds, tau, T1t, T1a, alpha, beta, lmbda, scale
    )

    print(f"Max absolute error: {max_err:.6e}")
    print(f"Relative error: {rel_err:.6e}")
    assert rel_err < 1e-5, f"Forward model relative error {rel_err} exceeds threshold 1e-5"


def test_forward_model_differentiability():
    plds = [1.525, 2.025, 2.525, 3.025]
    model = ASLForwardModel(plds)

    cbf = torch.tensor([40.0, 60.0], requires_grad=True)
    att = torch.tensor([1.2, 1.8], requires_grad=True)

    sig = model(cbf, att)
    assert sig.shape == (2, 4)
    assert torch.isfinite(sig).all()

    loss = sig.sum()
    loss.backward()

    assert cbf.grad is not None and torch.isfinite(cbf.grad).all()
    assert att.grad is not None and torch.isfinite(att.grad).all()
    assert (cbf.grad.abs() > 0).all()
    assert (att.grad.abs() > 0).all()


def test_forward_model_expected_observation():
    plds = [1.525, 2.025, 2.525, 3.025]
    model = ASLForwardModel(plds)

    cbf = torch.tensor([50.0])
    att = torch.tensor([1.5])

    clean = model(cbf, att)
    expected = model.expected_observation(cbf, att)

    # Expected observation should be 2 * clean
    diff = (expected - 2.0 * clean).abs().max().item()
    assert diff < 1e-5, f"Expected observation diff {diff} exceeds threshold"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

