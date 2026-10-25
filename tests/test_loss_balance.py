"""
test_loss_balance.py — Regression test for multi-task loss and gradient balance.

Verifies:
1. CBF and ATT gradient norms propagating through shared layers are balanced (ratio within 0.33 to 3.0),
   preventing the 43.8x gradient starvation defect identified in the diagnostic audit.
2. Physics consistency loss produces non-negligible gradient regularization (> 0.05).
"""

import sys
from pathlib import Path
import pytest
import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import build_model
from src.physics import ASLForwardModel
from src.losses import PCPLDNetLoss
from src.dataset import create_datasets


def test_rebalanced_gradient_parity():
    config_path = Path(__file__).resolve().parent.parent / 'configs' / 'default.yaml'
    with open(str(config_path), 'r') as f:
        cfg = yaml.safe_load(f)

    # Use rebalanced settings
    cfg['training']['regression_loss'] = 'L1'
    cfg['training']['att_loss_weight'] = 40.0
    cfg['training']['physics_loss_type'] = 'L1'
    cfg['training']['physics_lambda'] = 0.5

    cfg['dataset']['N_TRAIN'] = 2000
    cfg['dataset']['N_VAL'] = 500
    cfg['dataset']['N_TEST'] = 500

    data = create_datasets(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg, device)

    pcfg = cfg['physics']
    forward_model = ASLForwardModel(
        plds=pcfg['plds'], tau=pcfg['tau'], T1t=pcfg['T1t'],
        T1a=pcfg['T1a'], alpha=pcfg['alpha'], beta=pcfg['beta'],
        lmbda=pcfg['lmbda'], SCALE=pcfg['SCALE']
    ).to(device)

    criterion = PCPLDNetLoss(forward_model, data['X_mean'], data['X_std'], cfg).to(device)

    for xb, yb in data['train_loader']:
        xb, yb = xb.to(device), yb.to(device)
        break

    cbf_true, att_true = yb[:, 0], yb[:, 1]
    cbf_pred, att_pred = model(xb)

    # 1. CBF gradient
    model.zero_grad()
    l_cbf = criterion.cbf_loss_fn(cbf_pred, cbf_true)
    l_cbf.backward(retain_graph=True)
    g_cbf = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)).item()

    # 2. ATT gradient
    model.zero_grad()
    l_att = criterion.att_weight * criterion.att_loss_fn(att_pred, att_true)
    l_att.backward(retain_graph=True)
    g_att = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)).item()

    # 3. Physics gradient
    model.zero_grad()
    l_phys = criterion.physics_lambda * criterion.physics_loss(cbf_pred, att_pred, xb)
    l_phys.backward(retain_graph=True)
    g_phys = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)).item()

    ratio = g_cbf / (g_att + 1e-8)
    print(f"\nGradient check: CBF={g_cbf:.4f}, ATT={g_att:.4f}, Ratio={ratio:.2f}, Physics={g_phys:.4f}")

    # Assertions
    assert 0.33 <= ratio <= 3.0, f"CBF to ATT gradient ratio {ratio:.2f} is out of balanced range [0.33, 3.0]"
    assert g_phys > 0.05, f"Physics gradient norm {g_phys:.4f} is too weak (threshold 0.05)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

