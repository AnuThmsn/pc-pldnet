"""
measure_gradients_and_physics.py - Measures gradient norms with respect to shared trunk
and individual heads for CBF, ATT, and Physics losses, and analyzes Rician noise expectation.
"""

import sys
from pathlib import Path
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from scipy.special import iv

repo_root = Path("D:/ASL PARAMETER/pc-pldnet")
sys.path.insert(0, str(repo_root))

from src.model import build_model
from src.physics import ASLForwardModel
from src.losses import PCPLDNetLoss
from src.dataset import create_datasets

def rician_expectation(s, sigma):
    """Analytical expectation of Rician variable Rice(s, sigma)."""
    z = (s ** 2) / (2.0 * sigma ** 2 + 1e-15)
    # L_{1/2}(-z) = exp(z/2) * [(1+z)*I0(z/2) + z*I1(z/2)]
    return sigma * np.sqrt(np.pi / 2.0) * ((1.0 + z) * iv(0, z / 2.0) + z * iv(1, z / 2.0)) * np.exp(-z / 2.0)

def audit_dual_rician_bias(cfg):
    """Evaluate dual-Rician expectation vs 2*S across the test set SNR distribution."""
    print("Auditing Dual-Rician Observation Convention...")
    data = create_datasets(cfg)
    test_loader = data['test_loader']
    X_mean = data['X_mean']
    X_std = data['X_std']
    Y_test = data['Y_test']
    snr_test = data['snr_test']
    
    # We can also compute signals directly
    forward_model = ASLForwardModel(cfg['physics']['plds'], **{k: cfg['physics'][k] for k in ['tau', 'T1t', 'T1a', 'alpha', 'beta', 'lmbda', 'SCALE']})
    
    cbf_t = torch.tensor(Y_test[:5000, 0], dtype=torch.float32)
    att_t = torch.tensor(Y_test[:5000, 1], dtype=torch.float32)
    with torch.no_grad():
        clean_signals = forward_model(cbf_t, att_t).numpy()
    
    # Reconstruct raw noisy signals for these 5000 samples
    x_batch_list = []
    for x_b, _ in test_loader:
        x_batch_list.append(x_b)
        if sum(len(b) for b in x_batch_list) >= 5000:
            break
    x_norm = torch.cat(x_batch_list)[:5000].numpy()
    x_raw = x_norm * X_std + X_mean
    
    assumed_obs = 2.0 * clean_signals
    raw_obs = x_raw
    residual = raw_obs - assumed_obs
    
    snr_sub = snr_test[:5000]
    
    # Stratified Rician bias analysis
    snr_bins = [(0.0, 7.5), (7.5, 12.5), (12.5, 17.5), (17.5, 25.0), (25.0, 100.0)]
    bin_labels = ["< 7.5 (Low SNR)", "[7.5, 12.5) (~10)", "[12.5, 17.5) (~15)", "[17.5, 25.0) (~20)", ">= 25.0 (High SNR)"]
    
    bias_summary = []
    for (low, high), label in zip(snr_bins, bin_labels):
        m = (snr_sub >= low) & (snr_sub < high)
        if not np.any(m):
            continue
        mean_clean = float(np.mean(clean_signals[m]))
        mean_assumed = float(np.mean(assumed_obs[m]))
        mean_raw = float(np.mean(raw_obs[m]))
        mean_res = float(np.mean(residual[m]))
        rel_bias = float(mean_res / (mean_assumed + 1e-10))
        bias_summary.append({
            "snr_bin": label,
            "n_samples": int(np.sum(m)),
            "mean_clean_signal": mean_clean,
            "assumed_observation (2*S)": mean_assumed,
            "actual_mean_observation (mc+ml)": mean_raw,
            "rician_excess_bias (obs - 2*S)": mean_res,
            "relative_excess_percent": rel_bias * 100.0
        })
    
    return bias_summary

def measure_gradients(cfg_path, model_path=None):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg, device)
    
    if model_path and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded weights from {model_path}")
    else:
        print("Using randomly initialized weights for gradient audit")
    
    data = create_datasets(cfg)
    loader = data['train_loader']
    X_mean = data['X_mean']
    X_std = data['X_std']
    
    pcfg = cfg['physics']
    fwd = ASLForwardModel(pcfg['plds'], tau=pcfg['tau'], T1t=pcfg['T1t'],
                          T1a=pcfg['T1a'], alpha=pcfg['alpha'], beta=pcfg['beta'],
                          lmbda=pcfg['lmbda'], SCALE=pcfg['SCALE'])
    
    # We test both Original Loss and Rebalanced Loss formulations
    # Original: SmoothL1, w_att=2.0, phys_loss=SmoothL1, phys_lambda=0.01
    loss_orig = PCPLDNetLoss(fwd, X_mean, X_std, cfg).to(device)
    
    # Rebalanced: L1, w_att=40.0, phys_loss=L1, phys_lambda=0.5
    cfg_rebal = dict(cfg)
    cfg_rebal['training'] = dict(cfg['training'])
    cfg_rebal['training']['regression_loss'] = 'L1'
    cfg_rebal['training']['att_loss_weight'] = 40.0
    cfg_rebal['training']['physics_loss_type'] = 'L1'
    cfg_rebal['training']['physics_lambda'] = 0.5
    loss_rebal = PCPLDNetLoss(fwd, X_mean, X_std, cfg_rebal).to(device)

    # Measure over 10 batches
    shared_params = (list(model.encoder.parameters()) + 
                     list(model.conv_extractor.parameters()) + 
                     list(model.attn_proj.parameters()) + 
                     list(model.attention.parameters()) + 
                     list(model.shared_mlp.parameters()))
    cbf_params = list(model.cbf_head.parameters())
    att_params = list(model.att_head.parameters())
    
    def get_norm(params):
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return 0.0
        return torch.norm(torch.stack([torch.norm(g) for g in grads])).item()
    
    audit_results = {}
    for name, criterion, w_att, lam_phys in [
        ("Original_Objective (SmoothL1, w=2, lam=0.01)", loss_orig, 2.0, 0.01),
        ("Rebalanced_Objective (L1, w=40, lam=0.5)", loss_rebal, 40.0, 0.5)
    ]:
        trunk_cbf_norms, trunk_att_norms, trunk_phys_norms = [], [], []
        cbf_head_norms, att_head_norms = [], []
        cbf_losses, att_losses, phys_losses = [], [], []
        
        for i, (x_b, y_b) in enumerate(loader):
            if i >= 10:
                break
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            cbf_true, att_true = y_b[:, 0], y_b[:, 1]
            
            cbf_pred, att_pred = model(x_b)
            
            # 1. CBF term alone
            model.zero_grad()
            l_cbf = criterion.cbf_loss_fn(cbf_pred, cbf_true)
            l_cbf.backward(retain_graph=True)
            t_cbf = get_norm(shared_params)
            h_cbf = get_norm(cbf_params)
            
            # 2. ATT term alone
            model.zero_grad()
            l_att = criterion.att_loss_fn(att_pred, att_true) * w_att
            l_att.backward(retain_graph=True)
            t_att = get_norm(shared_params)
            h_att = get_norm(att_params)
            
            # 3. Physics term alone
            model.zero_grad()
            l_phys = criterion.physics_loss(cbf_pred, att_pred, x_b) * lam_phys
            l_phys.backward()
            t_phys = get_norm(shared_params)
            
            trunk_cbf_norms.append(t_cbf)
            trunk_att_norms.append(t_att)
            trunk_phys_norms.append(t_phys)
            cbf_head_norms.append(h_cbf)
            att_head_norms.append(h_att)
            
            cbf_losses.append(l_cbf.item())
            att_losses.append(l_att.item() / w_att)
            phys_losses.append(l_phys.item() / lam_phys)
            
        audit_results[name] = {
            "mean_loss_cbf": float(np.mean(cbf_losses)),
            "mean_loss_att_unweighted": float(np.mean(att_losses)),
            "mean_loss_physics_unweighted": float(np.mean(phys_losses)),
            "shared_trunk_grad_norm_CBF": float(np.mean(trunk_cbf_norms)),
            "shared_trunk_grad_norm_ATT": float(np.mean(trunk_att_norms)),
            "shared_trunk_grad_norm_Physics": float(np.mean(trunk_phys_norms)),
            "cbf_head_grad_norm": float(np.mean(cbf_head_norms)),
            "att_head_grad_norm": float(np.mean(att_head_norms)),
            "trunk_CBF_to_ATT_gradient_ratio": float(np.mean(trunk_cbf_norms) / (np.mean(trunk_att_norms) + 1e-10)),
            "trunk_Physics_to_Supervised_ratio": float(np.mean(trunk_phys_norms) / (np.mean(trunk_cbf_norms) + np.mean(trunk_att_norms) + 1e-10))
        }
    
    return audit_results

if __name__ == "__main__":
    with open(repo_root / "configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    bias_report = audit_dual_rician_bias(cfg)
    grad_report = measure_gradients(repo_root / "configs/default.yaml", repo_root / "models/final/best_model.pt")
    
    out = {
        "dual_rician_bias_analysis": bias_report,
        "gradient_norm_measurements": grad_report
    }
    
    with open(repo_root / "diagnostics/physics_validation_data.json", "w") as f:
        json.dump(out, f, indent=2)
    
    print("\n" + "="*80)
    print("DUAL-RICIAN OBSERVATION BIAS ANALYSIS (Observed vs 2*Clean)")
    print("="*80)
    for b in bias_report:
        print(f"{b['snr_bin']:<25} | N={b['n_samples']:<5} | Clean={b['mean_clean_signal']:6.1f} | 2*Clean={b['assumed_observation (2*S)']:6.1f} | Actual={b['actual_mean_observation (mc+ml)']:6.1f} | Excess Bias={b['rician_excess_bias (obs - 2*S)']:+6.1f} ({b['relative_excess_percent']:+5.1f}%)")
    
    print("\n" + "="*80)
    print("GRADIENT NORM MEASUREMENTS ON SHARED TRUNK AND HEADS")
    print("="*80)
    for k, v in grad_report.items():
        print(f"\nConfiguration: {k}")
        print(f"  Shared Trunk CBF Grad Norm:     {v['shared_trunk_grad_norm_CBF']:.4f}")
        print(f"  Shared Trunk ATT Grad Norm:     {v['shared_trunk_grad_norm_ATT']:.4f}")
        print(f"  Shared Trunk Physics Grad Norm: {v['shared_trunk_grad_norm_Physics']:.6f}")
        print(f"  CBF-to-ATT Gradient Ratio:      {v['trunk_CBF_to_ATT_gradient_ratio']:.2f} : 1")
        print(f"  Physics-to-Supervised Ratio:    {v['trunk_Physics_to_Supervised_ratio']:.4%}")
        print(f"  CBF Head Grad Norm:             {v['cbf_head_grad_norm']:.4f}")
        print(f"  ATT Head Grad Norm:             {v['att_head_grad_norm']:.4f}")
