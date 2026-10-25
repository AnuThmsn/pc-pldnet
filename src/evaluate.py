"""
evaluate.py — Final evaluation of PC-PLDNet on held-out test set.

Loads the best checkpoint, runs inference on the frozen test set,
computes all metrics (overall and by SNR), generates figures, and saves results.

Usage:
    python -m src.evaluate [--config configs/default.yaml]
"""

import os
import sys
import time
import json
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import build_model
from src.physics import ASLForwardModel
from src.losses import PCPLDNetLoss
from src.dataset import create_datasets
from src.metrics import (compute_all_metrics, compute_snr_stratified_metrics,
                          rmse, mae, mean_bias, r_squared,
                          concordance_correlation_coefficient)


def evaluate(cfg):
    project_dir = Path(__file__).resolve().parent.parent
    custom_eval_dir = cfg.get('evaluation', {}).get('output_dir', None)
    custom_model_dir = cfg.get('evaluation', {}).get('model_dir', None)
    if custom_eval_dir:
        results_dir = project_dir / custom_eval_dir
    else:
        results_dir = project_dir / 'experiments'
    if custom_model_dir:
        final_dir = project_dir / custom_model_dir
    else:
        final_dir = project_dir / 'models' / 'final'
    metrics_dir = results_dir / 'metrics'
    pred_dir = results_dir / 'predictions'
    fig_dir = results_dir / 'figures'

    for d in [metrics_dir, pred_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load best model
    best_path = final_dir / 'best_model.pt'
    if not best_path.exists():
        print(f"ERROR: Best model not found at {best_path}")
        sys.exit(1)

    ckpt = torch.load(str(best_path), map_location=device, weights_only=False)
    model = build_model(cfg, device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    n_params = model.count_parameters()

    print(f"Loaded best model from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")
    print(f"Parameters: {n_params:,}")

    # Generate test data
    data = create_datasets(cfg)
    test_loader = data['test_loader']
    Y_test = data['Y_test']
    snr_test = data['snr_test']

    # Inference
    print("Running inference on test set...")
    t_start = time.time()

    all_cbf_pred, all_att_pred = [], []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            cbf_pred, att_pred = model(x_batch)
            all_cbf_pred.append(cbf_pred.cpu().numpy())
            all_att_pred.append(att_pred.cpu().numpy())

    inference_time = time.time() - t_start
    cbf_pred = np.concatenate(all_cbf_pred)
    att_pred = np.concatenate(all_att_pred)
    cbf_true = Y_test[:, 0]
    att_true = Y_test[:, 1]

    print(f"Inference time: {inference_time:.3f}s for {len(cbf_pred)} samples")

    # Save predictions
    np.save(str(pred_dir / 'cbf_predictions.npy'), cbf_pred)
    np.save(str(pred_dir / 'att_predictions.npy'), att_pred)
    np.save(str(pred_dir / 'cbf_true.npy'), cbf_true)
    np.save(str(pred_dir / 'att_true.npy'), att_true)
    np.save(str(pred_dir / 'snr_test.npy'), snr_test)

    # Overall metrics
    cbf_metrics = compute_all_metrics(cbf_true, cbf_pred, 'CBF')
    att_metrics = compute_all_metrics(att_true, att_pred, 'ATT')
    overall = {**cbf_metrics, **att_metrics,
               'inference_time_s': inference_time,
               'n_params': n_params,
               'n_test': len(cbf_pred)}

    print(f"\n{'='*50}")
    print("OVERALL TEST METRICS")
    print(f"{'='*50}")
    print(f"CBF RMSE: {cbf_metrics['CBF_RMSE']:.4f} ml/100g/min")
    print(f"CBF MAE:  {cbf_metrics['CBF_MAE']:.4f} ml/100g/min")
    print(f"CBF Bias: {cbf_metrics['CBF_Bias']:.4f}")
    print(f"CBF R²:   {cbf_metrics['CBF_R2']:.4f}")
    print(f"CBF CCC:  {cbf_metrics['CBF_CCC']:.4f}")
    print(f"ATT RMSE: {att_metrics['ATT_RMSE']:.4f} s")
    print(f"ATT MAE:  {att_metrics['ATT_MAE']:.4f} s")
    print(f"ATT Bias: {att_metrics['ATT_Bias']:.4f}")
    print(f"ATT R²:   {att_metrics['ATT_R2']:.4f}")
    print(f"ATT CCC:  {att_metrics['ATT_CCC']:.4f}")

    # Save overall metrics
    with open(str(metrics_dir / 'overall_metrics.json'), 'w') as f:
        json.dump(overall, f, indent=2)

    overall_df = pd.DataFrame([overall])
    overall_df.to_csv(str(metrics_dir / 'overall_metrics.csv'), index=False)

    # SNR-stratified metrics
    # Define SNR bins centered on 10, 15, 20
    snr_bins = [
        (7.5, 12.5),   # ~SNR 10
        (12.5, 17.5),  # ~SNR 15
        (17.5, 25.0),  # ~SNR 20
    ]

    snr_results = compute_snr_stratified_metrics(
        cbf_true, cbf_pred, att_true, att_pred, snr_test, snr_bins
    )

    if snr_results:
        print(f"\n{'='*50}")
        print("SNR-STRATIFIED METRICS")
        print(f"{'='*50}")
        snr_df = pd.DataFrame(snr_results)
        print(snr_df.to_string(index=False))
        snr_df.to_csv(str(metrics_dir / 'snr_metrics.csv'), index=False)
    else:
        print("\nWARNING: No samples found in target SNR bins")

    # ── Generate figures ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 1. True vs Predicted scatter plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # CBF scatter
        ax = axes[0]
        ax.scatter(cbf_true, cbf_pred, s=1, alpha=0.1, c='steelblue')
        ax.plot([0, 100], [0, 100], 'r--', linewidth=1.5, label='Identity')
        ax.set_xlabel('True CBF (ml/100g/min)', fontsize=12)
        ax.set_ylabel('Predicted CBF (ml/100g/min)', fontsize=12)
        ax.set_title(f'CBF: RMSE={cbf_metrics["CBF_RMSE"]:.2f}, CCC={cbf_metrics["CBF_CCC"]:.3f}', fontsize=13)
        ax.legend()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')

        # ATT scatter
        ax = axes[1]
        ax.scatter(att_true, att_pred, s=1, alpha=0.1, c='coral')
        ax.plot([0.5, 3], [0.5, 3], 'r--', linewidth=1.5, label='Identity')
        ax.set_xlabel('True ATT (s)', fontsize=12)
        ax.set_ylabel('Predicted ATT (s)', fontsize=12)
        ax.set_title(f'ATT: RMSE={att_metrics["ATT_RMSE"]:.4f}s, CCC={att_metrics["ATT_CCC"]:.3f}', fontsize=13)
        ax.legend()
        ax.set_xlim(0.5, 3)
        ax.set_ylim(0.5, 3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(str(fig_dir / 'true_vs_predicted.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Error distribution histograms
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        cbf_errors = cbf_pred - cbf_true
        ax = axes[0]
        ax.hist(cbf_errors, bins=100, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax.set_xlabel('CBF Error (ml/100g/min)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'CBF Error Distribution (Bias={cbf_metrics["CBF_Bias"]:.3f})', fontsize=13)

        att_errors = att_pred - att_true
        ax = axes[1]
        ax.hist(att_errors, bins=100, color='coral', alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax.set_xlabel('ATT Error (s)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'ATT Error Distribution (Bias={att_metrics["ATT_Bias"]:.4f})', fontsize=13)

        plt.tight_layout()
        plt.savefig(str(fig_dir / 'error_distributions.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 3. Training curves (if history exists)
        history_path = project_dir / 'experiments' / 'logs' / 'training_history.json'
        if history_path.exists():
            with open(str(history_path), 'r') as f:
                history = json.load(f)

            epochs = [h['epoch'] for h in history]
            train_losses = [h['train_loss'] for h in history]
            val_losses = [h['val_loss'] for h in history]

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            ax = axes[0]
            ax.plot(epochs, train_losses, 'b-', label='Train', linewidth=1.5)
            ax.plot(epochs, val_losses, 'r-', label='Val', linewidth=1.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training & Validation Loss')
            ax.legend()
            ax.set_yscale('log')

            ax = axes[1]
            cbf_rmses = [h.get('CBF_RMSE', None) for h in history]
            ax.plot(epochs, cbf_rmses, 'b-', linewidth=1.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('CBF RMSE (ml/100g/min)')
            ax.set_title('Validation CBF RMSE')

            ax = axes[2]
            att_rmses = [h.get('ATT_RMSE', None) for h in history]
            ax.plot(epochs, att_rmses, 'r-', linewidth=1.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('ATT RMSE (s)')
            ax.set_title('Validation ATT RMSE')

            plt.tight_layout()
            plt.savefig(str(fig_dir / 'training_curves.png'), dpi=150, bbox_inches='tight')
            plt.close()

        print(f"\nFigures saved to {fig_dir}")

    except ImportError:
        print("WARNING: matplotlib not available, skipping figure generation")

    print(f"\nAll results saved to {results_dir}")
    return overall, snr_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate PC-PLDNet')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent.parent / args.config
    with open(str(config_path), 'r') as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg)


if __name__ == '__main__':
    main()

