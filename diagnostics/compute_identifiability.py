"""
compute_identifiability.py - Computes the analytical and numerical Jacobian of the
ASL forward model with respect to CBF and ATT, calculating singular values, condition
numbers, and collinearity metrics across representative parameter combinations.
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch

repo_root = Path("D:/ASL PARAMETER/pc-pldnet")
sys.path.insert(0, str(repo_root))

from src.physics import ASLForwardModel

def run_identifiability_analysis():
    plds = [1.525, 2.025, 2.525, 3.025]
    model = ASLForwardModel(plds)

    # Representative grid
    cbfs = [20.0, 50.0, 80.0]
    atts = [0.8, 1.2, 1.525, 1.8, 2.2, 2.6]

    # Coordinate scaling: normalized units [0, 1]
    # CBF_norm = CBF / 100.0, ATT_norm = (ATT - 0.5) / 2.5
    # Therefore dS/d(CBF_norm) = 100.0 * dS/d(CBF)
    # dS/d(ATT_norm) = 2.5 * dS/d(ATT)

    results = []
    numerical_deriv_check = []

    for cbf_val in cbfs:
        for att_val in atts:
            cbf_t = torch.tensor([cbf_val], dtype=torch.float64, requires_grad=True)
            att_t = torch.tensor([att_val], dtype=torch.float64, requires_grad=True)

            # Signal evaluation
            # Forward model in float64 for precision
            f_per_s = cbf_t / (6000.0 * model.lmbda)
            delta = att_t
            plds_t = torch.tensor(plds, dtype=torch.float64)

            prefix = 2.0 * model.alpha * model.beta * model.T1t * (1.0 / model.lmbda) * f_per_s
            e_att = torch.exp(-delta / model.T1a)
            t1 = torch.exp(-torch.clamp(plds_t - delta, min=0.0) / model.T1t)
            t2 = torch.exp(-torch.clamp(model.tau + plds_t - delta, min=0.0) / model.T1t)
            signals = prefix * e_att * (t1 - t2) * model.SCALE

            # Compute Jacobian: J is 4 x 2
            J_raw = np.zeros((4, 2))
            for i in range(4):
                if cbf_t.grad is not None:
                    cbf_t.grad.zero_()
                if att_t.grad is not None:
                    att_t.grad.zero_()
                signals[i].backward(retain_graph=True)
                J_raw[i, 0] = cbf_t.grad.item()
                J_raw[i, 1] = att_t.grad.item()

            # Scaled Jacobian (coordinates in [0, 1])
            J_scaled = np.zeros((4, 2))
            J_scaled[:, 0] = J_raw[:, 0] * 100.0   # dS / d(cbf_norm)
            J_scaled[:, 1] = J_raw[:, 1] * 2.5     # dS / d(att_norm)

            # SVD of scaled Jacobian
            U, S, Vt = np.linalg.svd(J_scaled, full_matrices=False)
            sigma1 = float(S[0])
            sigma2 = float(S[1])
            cond = float(sigma1 / (sigma2 + 1e-15)) if sigma2 > 1e-12 else float('inf')

            # Cosine similarity between columns of J_raw
            col_cbf = J_raw[:, 0]
            col_att = J_raw[:, 1]
            norm_cbf = np.linalg.norm(col_cbf)
            norm_att = np.linalg.norm(col_att)
            cos_sim = float(np.dot(col_cbf, col_att) / (norm_cbf * norm_att + 1e-15))
            angle_deg = float(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0))))

            # Theoretical gamma
            gamma = 1.0 / model.T1t - 1.0 / model.T1a
            expected_att_deriv = gamma * signals.detach().numpy()
            max_deriv_diff = float(np.max(np.abs(J_raw[:, 1] - expected_att_deriv))) if att_val <= 1.525 else None

            results.append({
                "cbf": float(cbf_val),
                "att": float(att_val),
                "is_sub_min_pld": bool(att_val <= 1.525),
                "signals": [float(x) for x in signals.detach().numpy()],
                "singular_values": [sigma1, sigma2],
                "condition_number": cond if cond != float('inf') else "inf",
                "cosine_similarity": cos_sim,
                "collinearity_angle_deg": angle_deg,
                "grad_norm_cbf_scaled": float(np.linalg.norm(J_scaled[:, 0])),
                "grad_norm_att_scaled": float(np.linalg.norm(J_scaled[:, 1])),
                "analytical_deriv_match_error": max_deriv_diff
            })

    # Target distribution statistics
    att_true = np.load(repo_root / "experiments/predictions/att_true.npy")
    p_orig = np.load(repo_root / "experiments/predictions/att_predictions.npy")
    p_rebal = np.load(repo_root / "diagnostics/rebalanced/experiments/predictions/att_predictions.npy")
    cbf_true = np.load(repo_root / "experiments/predictions/cbf_true.npy")
    p_cbf_orig = np.load(repo_root / "experiments/predictions/cbf_predictions.npy")
    p_cbf_rebal = np.load(repo_root / "diagnostics/rebalanced/experiments/predictions/cbf_predictions.npy")

    mask_early = att_true <= 1.525
    mask_late = ~mask_early

    target_stats = {
        "n_total": int(len(att_true)),
        "n_early_att": int(np.sum(mask_early)),
        "frac_early_att": float(np.mean(mask_early)),
        "theoretical_prob_early": float((1.525 - 0.5) / 2.5),
        "theoretical_mean_early": float((0.5 + 1.525) / 2.0),
        "observed_target_mean_early": float(np.mean(att_true[mask_early])),
        "observed_target_std_early": float(np.std(att_true[mask_early])),
        "observed_pred_mean_early_original": float(np.mean(p_orig[mask_early])),
        "observed_pred_mean_early_rebalanced": float(np.mean(p_rebal[mask_early])),
        "observed_pred_mean_late_original": float(np.mean(p_orig[mask_late])),
        "observed_pred_mean_late_rebalanced": float(np.mean(p_rebal[mask_late])),
        "gamma_theoretical": float(1.0 / model.T1t - 1.0 / model.T1a),
        "decay_ratio_consecutive_plds": float(np.exp(-0.5 / model.T1t))
    }

    output_data = {
        "description": "Rigorous Jacobian, SVD, and identifiability audit of the 4-PLD ASL forward model",
        "parameters": {
            "plds": plds,
            "tau": model.tau,
            "T1t": model.T1t,
            "T1a": model.T1a,
            "alpha": model.alpha,
            "beta": model.beta,
            "lmbda": model.lmbda,
            "SCALE": model.SCALE
        },
        "target_distribution_and_means": target_stats,
        "jacobian_grid_analysis": results
    }

    out_path = repo_root / "diagnostics/identifiability_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Successfully computed identifiability analysis and saved to {out_path}")

    # Print summary table
    print("\n" + "="*85)
    print(f"{'CBF':>6} | {'ATT':>6} | {'Regime':>10} | {'Sigma_1':>10} | {'Sigma_2':>10} | {'Cond Num':>10} | {'Cos Sim':>8} | {'Angle':>6}")
    print("="*85)
    for r in results:
        cond_str = f"{r['condition_number']:.1f}" if r['condition_number'] != "inf" else "INF"
        reg_str = "<= 1.525s" if r['is_sub_min_pld'] else "> 1.525s"
        print(f"{r['cbf']:6.1f} | {r['att']:6.3f} | {reg_str:>10} | {r['singular_values'][0]:10.2f} | {r['singular_values'][1]:10.4f} | {cond_str:>10} | {r['cosine_similarity']:8.5f} | {r['collinearity_angle_deg']:5.1f}°")

if __name__ == "__main__":
    run_identifiability_analysis()

