"""
run_ablation_pipeline.py - Trains missing ablation models (B, C, D), evaluates all 5
configurations (A, B, C, D, E) and DNN-G on the exact same 50,000 test set, computes
unified metrics with consistent SNR and ATT stratification, saves predictions, and
generates diagnostic comparison figures.
"""

import sys
from pathlib import Path
import json
import yaml
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo_root = Path("D:/ASL PARAMETER/pc-pldnet")
sys.path.insert(0, str(repo_root))

from src.train import train
from src.model import build_model
from src.dataset import create_datasets
from src.metrics import (compute_all_metrics, rmse, mae, mean_bias, r_squared,
                         concordance_correlation_coefficient)

ABLATIONS = [
    {
        "id": "A",
        "name": "A: Original (SmoothL1, w=2, phys=0.01 SmoothL1)",
        "config_path": repo_root / "configs/ablations/config_A.yaml",
        "model_path": repo_root / "diagnostics/ablations/A_orig/final/best_model.pt",
        "out_dir": repo_root / "diagnostics/ablations/A_orig",
        "history_path": repo_root / "experiments/logs/training_history.json"
    },
    {
        "id": "B",
        "name": "B: w_ATT=40 only (SmoothL1, w=40, phys=0.01 SmoothL1)",
        "config_path": repo_root / "configs/ablations/config_B.yaml",
        "model_path": repo_root / "diagnostics/ablations/B_smoothl1_att40/final/best_model.pt",
        "out_dir": repo_root / "diagnostics/ablations/B_smoothl1_att40",
        "history_path": repo_root / "diagnostics/ablations/B_smoothl1_att40/logs/training_history.json"
    },
    {
        "id": "C",
        "name": "C: L1 Regression only (L1, w=2, phys=0.01 SmoothL1)",
        "config_path": repo_root / "configs/ablations/config_C.yaml",
        "model_path": repo_root / "diagnostics/ablations/C_l1_att2/final/best_model.pt",
        "out_dir": repo_root / "diagnostics/ablations/C_l1_att2",
        "history_path": repo_root / "diagnostics/ablations/C_l1_att2/logs/training_history.json"
    },
    {
        "id": "D",
        "name": "D: L1 + w_ATT=40 (L1, w=40, orig phys=0.01 SmoothL1)",
        "config_path": repo_root / "configs/ablations/config_D.yaml",
        "model_path": repo_root / "diagnostics/ablations/D_l1_att40_origphys/final/best_model.pt",
        "out_dir": repo_root / "diagnostics/ablations/D_l1_att40_origphys",
        "history_path": repo_root / "diagnostics/ablations/D_l1_att40_origphys/logs/training_history.json"
    },
    {
        "id": "E",
        "name": "E: Full Rebalanced (L1, w=40, proposed phys=0.5 L1)",
        "config_path": repo_root / "configs/ablations/config_E.yaml",
        "model_path": repo_root / "diagnostics/ablations/E_rebalanced/final/best_model.pt",
        "out_dir": repo_root / "diagnostics/ablations/E_rebalanced",
        "history_path": repo_root / "diagnostics/rebalanced/logs/training_history.json"
    }
]

def train_missing_ablations():
    print("="*80)
    print("STEP 1: TRAINING MISSING ABLATIONS")
    print("="*80)
    for ab in ABLATIONS:
        if ab["model_path"].exists():
            print(f"[*] Found existing checkpoint for {ab['name']} at {ab['model_path']}")
        else:
            print(f"\n[>] Launching training for {ab['name']}...")
            with open(ab["config_path"], "r") as f:
                cfg = yaml.safe_load(f)
            t_start = time.time()
            train(cfg)
            print(f"[+] Completed training for {ab['name']} in {time.time() - t_start:.1f}s")

def evaluate_all():
    print("\n" + "="*80)
    print("STEP 2: EVALUATING ALL MODELS ON IDENTICAL 50,000 TEST SET")
    print("="*80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset using base config
    with open(repo_root / "configs/default.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)
    data = create_datasets(base_cfg)
    test_loader = data["test_loader"]
    Y_test = data["Y_test"]
    snr_test = data["snr_test"]
    cbf_true = Y_test[:, 0]
    att_true = Y_test[:, 1]
    
    # Stratification masks
    mask_early = att_true <= 1.525
    mask_late = att_true > 1.525
    
    snr_brackets = [
        ("< 7.5 (Low SNR)", (snr_test < 7.5)),
        ("[7.5, 12.5) (~10)", (snr_test >= 7.5) & (snr_test < 12.5)),
        ("[12.5, 17.5) (~15)", (snr_test >= 12.5) & (snr_test < 17.5)),
        ("[17.5, 25.0) (~20)", (snr_test >= 17.5) & (snr_test < 25.0)),
        (">= 25.0 (High SNR)", (snr_test >= 25.0))
    ]
    
    predictions_dict = {}
    
    # Load DNN-G predictions if available
    dnng_csv = repo_root / "experiments/metrics/baseline_comparison.csv"
    dnng_metrics_orig = None
    if dnng_csv.exists():
        df_base = pd.read_csv(dnng_csv)
        dnng_rows = df_base[df_base["Model"].str.contains("DNN-G")]
        if not dnng_rows.empty:
            dnng_metrics_orig = dnng_rows.iloc[0].to_dict()
    
    all_summary_rows = []
    
    for ab in ABLATIONS:
        print(f"\nEvaluating {ab['name']}...")
        with open(ab["config_path"], "r") as f:
            cfg = yaml.safe_load(f)
        
        model = build_model(cfg, device)
        ckpt = torch.load(ab["model_path"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        cbf_preds, att_preds = [], []
        with torch.no_grad():
            for x_b, _ in test_loader:
                x_b = x_b.to(device)
                cp, ap = model(x_b)
                cbf_preds.append(cp.cpu().numpy())
                att_preds.append(ap.cpu().numpy())
                
        cbf_p = np.concatenate(cbf_preds)
        att_p = np.concatenate(att_preds)
        
        predictions_dict[ab["id"]] = {"cbf": cbf_p, "att": att_p}
        
        # Save predictions in model out_dir
        pred_dir = ab["out_dir"] / "experiments/predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.save(pred_dir / "cbf_predictions.npy", cbf_p)
        np.save(pred_dir / "att_predictions.npy", att_p)
        
        # Gradient clipping frequency from history
        clip_freq = np.nan
        best_epoch = ckpt.get("epoch", np.nan)
        if ab["history_path"].exists():
            with open(ab["history_path"], "r") as f:
                hist = json.load(f)
            clip_rates = [h.get("clip_frequency", np.nan) for h in hist if "clip_frequency" in h]
            if clip_rates:
                clip_freq = float(np.mean(clip_rates))
        
        # Overall metrics
        cbf_m = compute_all_metrics(cbf_true, cbf_p, "CBF")
        att_m = compute_all_metrics(att_true, att_p, "ATT")
        
        # ATT stratified
        cbf_early_m = compute_all_metrics(cbf_true[mask_early], cbf_p[mask_early], "CBF")
        att_early_m = compute_all_metrics(att_true[mask_early], att_p[mask_early], "ATT")
        cbf_late_m = compute_all_metrics(cbf_true[mask_late], cbf_p[mask_late], "CBF")
        att_late_m = compute_all_metrics(att_true[mask_late], att_p[mask_late], "ATT")
        
        row = {
            "Ablation_ID": ab["id"],
            "Configuration": ab["name"],
            "Training_Samples": 300000,
            "Best_Epoch": best_epoch,
            "Mean_Clip_Frequency": clip_freq,
            
            # Overall
            "CBF_RMSE_overall": cbf_m["CBF_RMSE"],
            "CBF_MAE_overall": cbf_m["CBF_MAE"],
            "CBF_Bias_overall": cbf_m["CBF_Bias"],
            "CBF_R2_overall": cbf_m["CBF_R2"],
            "CBF_CCC_overall": cbf_m["CBF_CCC"],
            "ATT_RMSE_overall": att_m["ATT_RMSE"],
            "ATT_MAE_overall": att_m["ATT_MAE"],
            "ATT_Bias_overall": att_m["ATT_Bias"],
            "ATT_R2_overall": att_m["ATT_R2"],
            "ATT_CCC_overall": att_m["ATT_CCC"],
            
            # ATT <= 1.525s (Unidentifiable)
            "CBF_MAE_early": cbf_early_m["CBF_MAE"],
            "CBF_RMSE_early": cbf_early_m["CBF_RMSE"],
            "ATT_MAE_early": att_early_m["ATT_MAE"],
            "ATT_RMSE_early": att_early_m["ATT_RMSE"],
            "ATT_Bias_early": att_early_m["ATT_Bias"],
            "ATT_R2_early": att_early_m["ATT_R2"],
            
            # ATT > 1.525s (Identifiable)
            "CBF_MAE_late": cbf_late_m["CBF_MAE"],
            "CBF_RMSE_late": cbf_late_m["CBF_RMSE"],
            "ATT_MAE_late": att_late_m["ATT_MAE"],
            "ATT_RMSE_late": att_late_m["ATT_RMSE"],
            "ATT_Bias_late": att_late_m["ATT_Bias"],
            "ATT_R2_late": att_late_m["ATT_R2"],
        }
        
        # Add SNR-stratified columns
        for snr_label, snr_m in snr_brackets:
            c_s = compute_all_metrics(cbf_true[snr_m], cbf_p[snr_m], "CBF")
            a_s = compute_all_metrics(att_true[snr_m], att_p[snr_m], "ATT")
            row[f"CBF_RMSE_{snr_label}"] = c_s["CBF_RMSE"]
            row[f"CBF_MAE_{snr_label}"] = c_s["CBF_MAE"]
            row[f"ATT_RMSE_{snr_label}"] = a_s["ATT_RMSE"]
            row[f"ATT_MAE_{snr_label}"] = a_s["ATT_MAE"]
            
        all_summary_rows.append(row)
        
    # Add DNN-G reference row
    if dnng_metrics_orig:
        dnng_row = {
            "Ablation_ID": "DNN-G",
            "Configuration": "DNN-G Baseline (MLP, 2M samples)",
            "Training_Samples": 2000000,
            "Best_Epoch": "N/A",
            "Mean_Clip_Frequency": "N/A",
            "CBF_RMSE_overall": float(dnng_metrics_orig["CBF_RMSE"]),
            "CBF_MAE_overall": float(dnng_metrics_orig["CBF_MAE"]),
            "CBF_Bias_overall": float(dnng_metrics_orig["CBF_Bias"]),
            "CBF_R2_overall": float(dnng_metrics_orig["CBF_R2"]),
            "CBF_CCC_overall": float(dnng_metrics_orig["CBF_CCC"]),
            "ATT_RMSE_overall": float(dnng_metrics_orig["ATT_RMSE"]),
            "ATT_MAE_overall": float(dnng_metrics_orig["ATT_MAE"]),
            "ATT_Bias_overall": float(dnng_metrics_orig["ATT_Bias"]),
            "ATT_R2_overall": float(dnng_metrics_orig["ATT_R2"]),
            "ATT_CCC_overall": float(dnng_metrics_orig["ATT_CCC"]),
        }
        all_summary_rows.insert(0, dnng_row)
        
    # Save CSV
    summary_df = pd.DataFrame(all_summary_rows)
    out_csv = repo_root / "diagnostics/ablation_results.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved full ablation results table to {out_csv}")
    
    # Generate Diagnostic Comparison Figures
    generate_ablation_figures(predictions_dict, cbf_true, att_true, snr_test)

def generate_ablation_figures(predictions_dict, cbf_true, att_true, snr_test):
    fig_dir = repo_root / "diagnostics/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. ATT Error by Ablation and Identifiability Regime
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    model_ids = ["A", "B", "C", "D", "E"]
    labels = ["A: Orig", "B: w=40", "C: L1", "D: L1+w40", "E: Rebal"]
    
    mask_early = att_true <= 1.525
    mask_late = att_true > 1.525
    
    early_maes = [float(np.mean(np.abs(predictions_dict[m]["att"][mask_early] - att_true[mask_early]))) for m in model_ids]
    late_maes = [float(np.mean(np.abs(predictions_dict[m]["att"][mask_late] - att_true[mask_late]))) for m in model_ids]
    
    x = np.arange(len(model_ids))
    width = 0.35
    
    ax = axes[0]
    ax.bar(x - width/2, early_maes, width, label='Unidentifiable (ATT <= 1.525s)', color='#d62728', alpha=0.8)
    ax.bar(x + width/2, late_maes, width, label='Identifiable (ATT > 1.525s)', color='#2ca02c', alpha=0.8)
    ax.set_ylabel('ATT MAE (s)')
    ax.set_title('ATT Estimation MAE Across Ablations by Physical Regime')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # CBF MAE
    cbf_early_maes = [float(np.mean(np.abs(predictions_dict[m]["cbf"][mask_early] - cbf_true[mask_early]))) for m in model_ids]
    cbf_late_maes = [float(np.mean(np.abs(predictions_dict[m]["cbf"][mask_late] - cbf_true[mask_late]))) for m in model_ids]
    
    ax = axes[1]
    ax.bar(x - width/2, cbf_early_maes, width, label='Unidentifiable (ATT <= 1.525s)', color='#d62728', alpha=0.8)
    ax.bar(x + width/2, cbf_late_maes, width, label='Identifiable (ATT > 1.525s)', color='#2ca02c', alpha=0.8)
    ax.set_ylabel('CBF MAE (ml/100g/min)')
    ax.set_title('CBF Estimation MAE Across Ablations by Physical Regime')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(fig_dir / "ablation_regime_breakdown.png", dpi=200)
    plt.close()
    
    # 2. CBF-ATT Residual Coupling (A vs D vs E)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sub_idx = np.random.RandomState(42).choice(len(cbf_true), 4000, replace=False)
    
    for ax, mid, title in zip(axes, ["A", "D", "E"], ["A: Original Baseline", "D: L1 + w=40 (Orig Phys)", "E: Full Rebalanced"]):
        c_res = predictions_dict[mid]["cbf"][sub_idx] - cbf_true[sub_idx]
        a_res = predictions_dict[mid]["att"][sub_idx] - att_true[sub_idx]
        corr = np.corrcoef(c_res, a_res)[0, 1]
        
        sc = ax.scatter(a_res, c_res, alpha=0.15, s=6, c=snr_test[sub_idx], cmap='viridis', vmin=5, vmax=30)
        ax.axhline(0, color='k', linestyle=':', lw=1)
        ax.axvline(0, color='k', linestyle=':', lw=1)
        ax.set_title(f"{title}\nResidual Corr: r = {corr:.3f}")
        ax.set_xlabel("ATT Residual (Pred - True) [s]")
        ax.set_ylabel("CBF Residual (Pred - True)")
        ax.grid(True, alpha=0.3)
        if mid == "E":
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("True Sample SNR")
            
    plt.tight_layout()
    plt.savefig(fig_dir / "residual_coupling_ablations.png", dpi=200)
    plt.close()
    
    print(f"[+] Saved diagnostic comparison plots to {fig_dir}")

if __name__ == "__main__":
    train_missing_ablations()
    evaluate_all()

