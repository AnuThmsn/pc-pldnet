"""
evaluate_and_compare.py - Evaluates the corrected/rebalanced PC-PLDNet checkpoint
and compares it directly against the baseline DNN-G and original PC-PLDNet.
"""

import sys
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo_root = Path("D:/ASL PARAMETER/pc-pldnet")
sys.path.insert(0, str(repo_root))

from src.model import build_model
from src.dataset import create_datasets
from src.metrics import (compute_all_metrics, compute_snr_stratified_metrics,
                         rmse, mae, mean_bias, r_squared,
                         concordance_correlation_coefficient)

def run_evaluation_and_comparison():
    config_path = repo_root / "configs/rebalanced.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    best_model_path = repo_root / "diagnostics/rebalanced/final/best_model.pt"
    if not best_model_path.exists():
        print(f"ERROR: Model checkpoint not found at {best_model_path}")
        return

    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    print(f"Loaded corrected checkpoint from epoch {ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.6f}")

    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Generate exact test dataset
    data = create_datasets(cfg)
    test_loader = data["test_loader"]
    Y_test = data["Y_test"]
    snr_test = data["snr_test"]
    cbf_true = Y_test[:, 0]
    att_true = Y_test[:, 1]

    # Run inference
    all_cbf_pred, all_att_pred = [], []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            cbf_p, att_p = model(x_batch)
            all_cbf_pred.append(cbf_p.cpu().numpy())
            all_att_pred.append(att_p.cpu().numpy())

    cbf_pred_corr = np.concatenate(all_cbf_pred)
    att_pred_corr = np.concatenate(all_att_pred)

    # Save corrected predictions
    pred_dir = repo_root / "diagnostics/rebalanced/experiments/predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "cbf_predictions.npy", cbf_pred_corr)
    np.save(pred_dir / "att_predictions.npy", att_pred_corr)
    np.save(pred_dir / "cbf_true.npy", cbf_true)
    np.save(pred_dir / "att_true.npy", att_true)
    np.save(pred_dir / "snr_test.npy", snr_test)

    # Load original PC-PLDNet predictions
    orig_cbf_pred = np.load(repo_root / "experiments/predictions/cbf_predictions.npy")
    orig_att_pred = np.load(repo_root / "experiments/predictions/att_predictions.npy")

    # Load DNN-G baseline numbers
    baseline_df = pd.read_csv(repo_root / "experiments/metrics/baseline_comparison.csv")
    dnng_row = baseline_df[baseline_df["Model"].str.contains("DNN-G")].iloc[0]

    # Compute overall metrics for Corrected PC-PLDNet
    cbf_corr_metrics = compute_all_metrics(cbf_true, cbf_pred_corr, "CBF")
    att_corr_metrics = compute_all_metrics(att_true, att_pred_corr, "ATT")

    cbf_orig_metrics = compute_all_metrics(cbf_true, orig_cbf_pred, "CBF")
    att_orig_metrics = compute_all_metrics(att_true, orig_att_pred, "ATT")

    # Overall comparison table
    comparison_rows = [
        {
            "Model": "DNN-G (Baseline)",
            "Samples": "2,000,000",
            "CBF_RMSE": float(dnng_row["CBF_RMSE"]),
            "CBF_MAE": float(dnng_row["CBF_MAE"]),
            "CBF_Bias": float(dnng_row["CBF_Bias"]),
            "CBF_R2": float(dnng_row["CBF_R2"]),
            "CBF_CCC": float(dnng_row["CBF_CCC"]),
            "ATT_RMSE": float(dnng_row["ATT_RMSE"]),
            "ATT_MAE": float(dnng_row["ATT_MAE"]),
            "ATT_Bias": float(dnng_row["ATT_Bias"]),
            "ATT_R2": float(dnng_row["ATT_R2"]),
            "ATT_CCC": float(dnng_row["ATT_CCC"]),
        },
        {
            "Model": "PC-PLDNet (Original)",
            "Samples": "300,000",
            "CBF_RMSE": float(cbf_orig_metrics["CBF_RMSE"]),
            "CBF_MAE": float(cbf_orig_metrics["CBF_MAE"]),
            "CBF_Bias": float(cbf_orig_metrics["CBF_Bias"]),
            "CBF_R2": float(cbf_orig_metrics["CBF_R2"]),
            "CBF_CCC": float(cbf_orig_metrics["CBF_CCC"]),
            "ATT_RMSE": float(att_orig_metrics["ATT_RMSE"]),
            "ATT_MAE": float(att_orig_metrics["ATT_MAE"]),
            "ATT_Bias": float(att_orig_metrics["ATT_Bias"]),
            "ATT_R2": float(att_orig_metrics["ATT_R2"]),
            "ATT_CCC": float(att_orig_metrics["ATT_CCC"]),
        },
        {
            "Model": "PC-PLDNet (Rebalanced)",
            "Samples": "300,000",
            "CBF_RMSE": float(cbf_corr_metrics["CBF_RMSE"]),
            "CBF_MAE": float(cbf_corr_metrics["CBF_MAE"]),
            "CBF_Bias": float(cbf_corr_metrics["CBF_Bias"]),
            "CBF_R2": float(cbf_corr_metrics["CBF_R2"]),
            "CBF_CCC": float(cbf_corr_metrics["CBF_CCC"]),
            "ATT_RMSE": float(att_corr_metrics["ATT_RMSE"]),
            "ATT_MAE": float(att_corr_metrics["ATT_MAE"]),
            "ATT_Bias": float(att_corr_metrics["ATT_Bias"]),
            "ATT_R2": float(att_corr_metrics["ATT_R2"]),
            "ATT_CCC": float(att_corr_metrics["ATT_CCC"]),
        }
    ]

    comp_df = pd.DataFrame(comparison_rows)
    metrics_dir = repo_root / "diagnostics/rebalanced/experiments/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(metrics_dir / "baseline_comparison.csv", index=False)
    print("\n" + "="*80)
    print("OVERALL MODEL COMPARISON (50,000 test samples)")
    print("="*80)
    print(comp_df.to_string(index=False))

    # SNR Stratification
    snr_bins = [(7.5, 12.5), (12.5, 17.5), (17.5, 25.0)]
    snr_labels = ["[8, 12)", "[12, 18)", "[18, 25)"]
    snr_comp_rows = []

    # Load baseline SNR comparison
    base_snr_df = pd.read_csv(repo_root / "experiments/metrics/snr_comparison.csv")
    for row in base_snr_df[base_snr_df["Model"] == "DNN-G"].to_dict("records"):
        snr_comp_rows.append(row)
    for row in base_snr_df[base_snr_df["Model"] == "PC-PLDNet"].to_dict("records"):
        row["Model"] = "PC-PLDNet (Original)"
        snr_comp_rows.append(row)

    # Compute for Corrected
    for (low, high), label in zip(snr_bins, snr_labels):
        mask = (snr_test >= low) & (snr_test < high)
        n = int(np.sum(mask))
        c_t, c_p = cbf_true[mask], cbf_pred_corr[mask]
        a_t, a_p = att_true[mask], att_pred_corr[mask]
        c_m = compute_all_metrics(c_t, c_p, "CBF")
        a_m = compute_all_metrics(a_t, a_p, "ATT")
        snr_comp_rows.append({
            "SNR_range": label,
            "N_samples": n,
            "CBF_RMSE": c_m["CBF_RMSE"],
            "CBF_MAE": c_m["CBF_MAE"],
            "CBF_Bias": c_m["CBF_Bias"],
            "CBF_R2": c_m["CBF_R2"],
            "CBF_Pearson_r": c_m.get("CBF_Pearson_r", np.nan),
            "CBF_CCC": c_m["CBF_CCC"],
            "ATT_RMSE": a_m["ATT_RMSE"],
            "ATT_MAE": a_m["ATT_MAE"],
            "ATT_Bias": a_m["ATT_Bias"],
            "ATT_R2": a_m["ATT_R2"],
            "ATT_Pearson_r": a_m.get("ATT_Pearson_r", np.nan),
            "ATT_CCC": a_m["ATT_CCC"],
            "Model": "PC-PLDNet (Rebalanced)"
        })

    all_snr_df = pd.DataFrame(snr_comp_rows)
    all_snr_df.to_csv(metrics_dir / "snr_comparison.csv", index=False)
    print("\n" + "="*80)
    print("SNR-STRATIFIED COMPARISON")
    print("="*80)
    print(all_snr_df[["SNR_range", "Model", "CBF_RMSE", "CBF_MAE", "ATT_RMSE", "ATT_MAE"]].to_string(index=False))

    # Identifiability Regime Comparison (ATT <= 1.525 vs ATT > 1.525)
    id_mask = att_true > 1.525
    non_id_mask = ~id_mask

    regime_rows = [
        {
            "Regime": "Identifiable (ATT > 1.525s, N=29,484)",
            "Model": "PC-PLDNet (Original)",
            "CBF_MAE": float(mae(cbf_true[id_mask], orig_cbf_pred[id_mask])),
            "CBF_RMSE": float(rmse(cbf_true[id_mask], orig_cbf_pred[id_mask])),
            "ATT_MAE": float(mae(att_true[id_mask], orig_att_pred[id_mask])),
            "ATT_RMSE": float(rmse(att_true[id_mask], orig_att_pred[id_mask])),
            "ATT_Bias": float(mean_bias(att_true[id_mask], orig_att_pred[id_mask])),
            "ATT_R2": float(r_squared(att_true[id_mask], orig_att_pred[id_mask])),
        },
        {
            "Regime": "Identifiable (ATT > 1.525s, N=29,484)",
            "Model": "PC-PLDNet (Rebalanced)",
            "CBF_MAE": float(mae(cbf_true[id_mask], cbf_pred_corr[id_mask])),
            "CBF_RMSE": float(rmse(cbf_true[id_mask], cbf_pred_corr[id_mask])),
            "ATT_MAE": float(mae(att_true[id_mask], att_pred_corr[id_mask])),
            "ATT_RMSE": float(rmse(att_true[id_mask], att_pred_corr[id_mask])),
            "ATT_Bias": float(mean_bias(att_true[id_mask], att_pred_corr[id_mask])),
            "ATT_R2": float(r_squared(att_true[id_mask], att_pred_corr[id_mask])),
        },
        {
            "Regime": "Unidentifiable (ATT <= 1.525s, N=20,516)",
            "Model": "PC-PLDNet (Original)",
            "CBF_MAE": float(mae(cbf_true[non_id_mask], orig_cbf_pred[non_id_mask])),
            "CBF_RMSE": float(rmse(cbf_true[non_id_mask], orig_cbf_pred[non_id_mask])),
            "ATT_MAE": float(mae(att_true[non_id_mask], orig_att_pred[non_id_mask])),
            "ATT_RMSE": float(rmse(att_true[non_id_mask], orig_att_pred[non_id_mask])),
            "ATT_Bias": float(mean_bias(att_true[non_id_mask], orig_att_pred[non_id_mask])),
            "ATT_R2": float(r_squared(att_true[non_id_mask], orig_att_pred[non_id_mask])),
        },
        {
            "Regime": "Unidentifiable (ATT <= 1.525s, N=20,516)",
            "Model": "PC-PLDNet (Rebalanced)",
            "CBF_MAE": float(mae(cbf_true[non_id_mask], cbf_pred_corr[non_id_mask])),
            "CBF_RMSE": float(rmse(cbf_true[non_id_mask], cbf_pred_corr[non_id_mask])),
            "ATT_MAE": float(mae(att_true[non_id_mask], att_pred_corr[non_id_mask])),
            "ATT_RMSE": float(rmse(att_true[non_id_mask], att_pred_corr[non_id_mask])),
            "ATT_Bias": float(mean_bias(att_true[non_id_mask], att_pred_corr[non_id_mask])),
            "ATT_R2": float(r_squared(att_true[non_id_mask], att_pred_corr[non_id_mask])),
        }
    ]
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(metrics_dir / "regime_comparison.csv", index=False)
    print("\n" + "="*80)
    print("PHYSICAL IDENTIFIABILITY REGIME COMPARISON")
    print("="*80)
    print(regime_df.to_string(index=False))

    # Generate Figure: Comparison of True vs Predicted and Error Distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # CBF True vs Pred (Rebalanced)
    idx_sub = np.random.RandomState(42).choice(len(cbf_true), 5000, replace=False)
    ax = axes[0, 0]
    ax.scatter(cbf_true[idx_sub], cbf_pred_corr[idx_sub], alpha=0.2, s=8, c='#1f77b4', label='Rebalanced')
    ax.plot([0, 100], [0, 100], 'k--', lw=1.5, label='Identity')
    ax.set_title(f"CBF True vs Predicted (Rebalanced)\nRMSE={cbf_corr_metrics['CBF_RMSE']:.3f}, R2={cbf_corr_metrics['CBF_R2']:.4f}")
    ax.set_xlabel("True CBF (ml/100g/min)")
    ax.set_ylabel("Predicted CBF (ml/100g/min)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ATT True vs Pred (Rebalanced)
    ax = axes[0, 1]
    ax.scatter(att_true[idx_sub], att_pred_corr[idx_sub], alpha=0.2, s=8, c='#ff7f0e', label='Rebalanced')
    ax.axvline(1.525, color='red', linestyle=':', lw=1.5, label='Min PLD (1.525s)')
    ax.plot([0.5, 3.0], [0.5, 3.0], 'k--', lw=1.5, label='Identity')
    ax.set_title(f"ATT True vs Predicted (Rebalanced)\nRMSE={att_corr_metrics['ATT_RMSE']:.3f}s, MAE={att_corr_metrics['ATT_MAE']:.3f}s")
    ax.set_xlabel("True ATT (s)")
    ax.set_ylabel("Predicted ATT (s)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # CBF Error Comparison (Original vs Rebalanced)
    ax = axes[1, 0]
    err_cbf_orig = np.abs(orig_cbf_pred - cbf_true)
    err_cbf_corr = np.abs(cbf_pred_corr - cbf_true)
    ax.hist(err_cbf_orig, bins=50, range=(0, 15), alpha=0.5, density=True, label=f"Original (MAE={np.mean(err_cbf_orig):.3f})", color='grey')
    ax.hist(err_cbf_corr, bins=50, range=(0, 15), alpha=0.5, density=True, label=f"Rebalanced (MAE={np.mean(err_cbf_corr):.3f})", color='#1f77b4')
    ax.set_title("CBF Absolute Error Distribution")
    ax.set_xlabel("|Predicted CBF - True CBF| (ml/100g/min)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ATT Error Comparison (Original vs Rebalanced)
    ax = axes[1, 1]
    err_att_orig = np.abs(orig_att_pred - att_true)
    err_att_corr = np.abs(att_pred_corr - att_true)
    ax.hist(err_att_orig, bins=50, range=(0, 1.5), alpha=0.5, density=True, label=f"Original (MAE={np.mean(err_att_orig):.3f}s)", color='grey')
    ax.hist(err_att_corr, bins=50, range=(0, 1.5), alpha=0.5, density=True, label=f"Rebalanced (MAE={np.mean(err_att_corr):.3f}s)", color='#ff7f0e')
    ax.set_title("ATT Absolute Error Distribution")
    ax.set_xlabel("|Predicted ATT - True ATT| (s)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig_path = repo_root / "diagnostics/figures/rebalanced_performance_comparison.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"\nSaved comparison figure to {fig_path}")

if __name__ == "__main__":
    run_evaluation_and_comparison()

