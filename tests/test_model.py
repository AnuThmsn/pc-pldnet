"""
test_model.py — Automated tests for PC-PLDNet architecture.
"""

import sys
import tempfile
from pathlib import Path
import pytest
import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import build_model, PCPLDNet
from src.physics import ASLForwardModel
from src.losses import PCPLDNetLoss


@pytest.fixture
def sample_config():
    config_path = Path(__file__).resolve().parent.parent / 'configs' / 'default.yaml'
    with open(str(config_path), 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def test_model_input_output_shapes(sample_config):
    model = build_model(sample_config, device='cpu')
    model.eval()

    batch_size = 16
    x = torch.randn(batch_size, 4)

    with torch.no_grad():
        cbf_pred, att_pred = model(x)

    assert cbf_pred.shape == (batch_size,), f"Expected shape ({batch_size},), got {cbf_pred.shape}"
    assert att_pred.shape == (batch_size,), f"Expected shape ({batch_size},), got {att_pred.shape}"


def test_model_finite_outputs(sample_config):
    model = build_model(sample_config, device='cpu')
    model.eval()

    # Test with standard inputs, extreme inputs, and zeros
    inputs = [
        torch.randn(32, 4),
        torch.zeros(16, 4),
        torch.ones(16, 4) * 10.0,
        torch.ones(16, 4) * -10.0,
    ]

    for x in inputs:
        with torch.no_grad():
            cbf_pred, att_pred = model(x)
        assert torch.isfinite(cbf_pred).all(), "CBF predictions contain NaN or Inf"
        assert torch.isfinite(att_pred).all(), "ATT predictions contain NaN or Inf"


def test_model_output_ranges(sample_config):
    model = build_model(sample_config, device='cpu')
    model.eval()

    cbf_min = sample_config['targets']['CBF_MIN']
    cbf_max = sample_config['targets']['CBF_MAX']
    att_min = sample_config['targets']['ATT_MIN']
    att_max = sample_config['targets']['ATT_MAX']

    # Extreme inputs to push outputs towards bounds
    x = torch.randn(100, 4) * 50.0

    with torch.no_grad():
        cbf_pred, att_pred = model(x)

    assert (cbf_pred >= cbf_min).all(), f"CBF pred below {cbf_min}: min was {cbf_pred.min().item()}"
    assert (cbf_pred <= cbf_max).all(), f"CBF pred above {cbf_max}: max was {cbf_pred.max().item()}"
    assert (att_pred >= att_min).all(), f"ATT pred below {att_min}: min was {att_pred.min().item()}"
    assert (att_pred <= att_max).all(), f"ATT pred above {att_max}: max was {att_pred.max().item()}"


def test_model_loss_and_gradients(sample_config):
    model = build_model(sample_config, device='cpu')
    model.train()

    pcfg = sample_config['physics']
    forward_model = ASLForwardModel(
        plds=pcfg['plds'], tau=pcfg['tau'], T1t=pcfg['T1t'],
        T1a=pcfg['T1a'], alpha=pcfg['alpha'], beta=pcfg['beta'],
        lmbda=pcfg['lmbda'], SCALE=pcfg['SCALE']
    )

    X_mean = torch.tensor([[662.0, 566.0, 442.0, 317.0]])
    X_std = torch.tensor([[476.0, 341.0, 250.0, 176.0]])

    criterion = PCPLDNetLoss(forward_model, X_mean, X_std, sample_config)

    batch_size = 8
    x = torch.randn(batch_size, 4)
    cbf_true = torch.empty(batch_size).uniform_(10.0, 90.0)
    att_true = torch.empty(batch_size).uniform_(0.8, 2.5)

    cbf_pred, att_pred = model(x)
    total_loss, ldict = criterion(cbf_pred, att_pred, cbf_true, att_true, x)

    assert torch.isfinite(total_loss), f"Total loss is not finite: {total_loss.item()}"
    for k, v in ldict.items():
        assert not (v != v), f"Loss component {k} is NaN"

    total_loss.backward()

    # Check gradients exist and are finite on key modules
    grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
            grad_count += 1
    assert grad_count > 0, "No trainable parameters received gradients"


def test_checkpoint_saving_and_loading(sample_config):
    model1 = build_model(sample_config, device='cpu')
    model1.eval()

    x = torch.randn(10, 4)
    with torch.no_grad():
        cbf1, att1 = model1(x)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torch.save({'model_state_dict': model1.state_dict()}, tmp_path)

        model2 = build_model(sample_config, device='cpu')
        model2.eval()
        ckpt = torch.load(tmp_path, map_location='cpu', weights_only=False)
        model2.load_state_dict(ckpt['model_state_dict'])

        with torch.no_grad():
            cbf2, att2 = model2(x)

        assert torch.allclose(cbf1, cbf2, atol=1e-6), "Loaded model CBF outputs differ"
        assert torch.allclose(att1, att2, atol=1e-6), "Loaded model ATT outputs differ"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

