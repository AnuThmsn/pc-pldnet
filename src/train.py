"""
train.py — Training script for PC-PLDNet.

Usage:
    python -m src.train [--config configs/default.yaml] [--resume]
"""

import os
import sys
import time
import json
import copy
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import build_model, PCPLDNet
from src.physics import ASLForwardModel
from src.losses import PCPLDNetLoss
from src.dataset import create_datasets
from src.metrics import compute_all_metrics


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    loss_components = {'L_CBF': 0, 'L_ATT': 0, 'L_physics': 0, 'L_total': 0}
    n_batches = 0
    n_clipped = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        cbf_true = y_batch[:, 0].to(device, non_blocking=True)
        att_true = y_batch[:, 1].to(device, non_blocking=True)

        optimizer.zero_grad()
        cbf_pred, att_pred = model(x_batch)
        loss, ldict = criterion(cbf_pred, att_pred, cbf_true, att_true, x_batch)

        loss.backward()
        if grad_clip > 0:
            norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if norm > grad_clip:
                n_clipped += 1
        optimizer.step()

        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += ldict[k]
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    for k in loss_components:
        loss_components[k] /= max(n_batches, 1)
    clip_freq = n_clipped / max(n_batches, 1)
    return avg_loss, loss_components, clip_freq


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    loss_components = {'L_CBF': 0, 'L_ATT': 0, 'L_physics': 0, 'L_total': 0}
    n_batches = 0

    all_cbf_pred, all_att_pred = [], []
    all_cbf_true, all_att_true = [], []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        cbf_true = y_batch[:, 0].to(device, non_blocking=True)
        att_true = y_batch[:, 1].to(device, non_blocking=True)

        cbf_pred, att_pred = model(x_batch)
        loss, ldict = criterion(cbf_pred, att_pred, cbf_true, att_true, x_batch)

        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += ldict[k]
        n_batches += 1

        all_cbf_pred.append(cbf_pred.cpu().numpy())
        all_att_pred.append(att_pred.cpu().numpy())
        all_cbf_true.append(cbf_true.cpu().numpy())
        all_att_true.append(att_true.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    for k in loss_components:
        loss_components[k] /= max(n_batches, 1)

    cbf_pred = np.concatenate(all_cbf_pred)
    att_pred = np.concatenate(all_att_pred)
    cbf_true = np.concatenate(all_cbf_true)
    att_true = np.concatenate(all_att_true)

    cbf_metrics = compute_all_metrics(cbf_true, cbf_pred, 'CBF')
    att_metrics = compute_all_metrics(att_true, att_pred, 'ATT')

    return avg_loss, loss_components, {**cbf_metrics, **att_metrics}


def train(cfg, resume=False):
    """Main training function."""
    project_dir = Path(__file__).resolve().parent.parent
    custom_dir = cfg.get('training', {}).get('output_dir', None)
    if custom_dir:
        base_out = project_dir / custom_dir
        ckpt_dir = base_out / 'checkpoints'
        final_dir = base_out / 'final'
        log_dir = base_out / 'logs'
    else:
        ckpt_dir = project_dir / 'models' / 'checkpoints'
        final_dir = project_dir / 'models' / 'final'
        log_dir = project_dir / 'experiments' / 'logs'

    for d in [ckpt_dir, final_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Set seed
    set_seed(cfg['training']['seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create datasets
    data = create_datasets(cfg)
    train_loader = data['train_loader']
    val_loader = data['val_loader']

    # Save normalization stats
    np.save(str(final_dir / 'X_mean.npy'), data['X_mean'])
    np.save(str(final_dir / 'X_std.npy'), data['X_std'])

    # Build model
    model = build_model(cfg, device)
    n_params = model.count_parameters()
    print(f"PC-PLDNet parameters: {n_params:,}")

    # Build forward model and loss
    pcfg = cfg['physics']
    forward_model = ASLForwardModel(
        plds=pcfg['plds'], tau=pcfg['tau'], T1t=pcfg['T1t'],
        T1a=pcfg['T1a'], alpha=pcfg['alpha'], beta=pcfg['beta'],
        lmbda=pcfg['lmbda'], SCALE=pcfg['SCALE']
    ).to(device)

    criterion = PCPLDNetLoss(forward_model, data['X_mean'], data['X_std'], cfg).to(device)

    # Optimizer
    tcfg = cfg['training']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg['lr'],
        weight_decay=tcfg['weight_decay']
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=tcfg['scheduler_factor'],
        patience=tcfg['scheduler_patience'],
        min_lr=tcfg['scheduler_min_lr'],
    )

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    resume_path = ckpt_dir / 'latest_checkpoint.pt'
    if resume and resume_path.exists():
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        patience_counter = ckpt['patience_counter']
        history = ckpt.get('history', [])
        print(f"  Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training: {tcfg['max_epochs']} epochs, batch_size={tcfg['batch_size']}")
    print(f"Train: {data['N_TRAIN']} | Val: {data['N_VAL']} | Test: {data['N_TEST']}")
    print(f"{'='*60}\n")

    train_start = time.time()

    for epoch in range(start_epoch, tcfg['max_epochs']):
        epoch_start = time.time()

        # Train
        train_loss, train_components, clip_freq = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=tcfg['grad_clip']
        )

        # Validate
        val_loss, val_components, val_metrics = validate(
            model, val_loader, criterion, device
        )

        # LR scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start

        # Log
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr,
            'epoch_time_s': epoch_time,
            'clip_frequency': clip_freq,
            **{f'train_{k}': v for k, v in train_components.items()},
            **{f'val_{k}': v for k, v in val_components.items()},
            **val_metrics,
        }
        history.append(log_entry)

        # Print progress
        print(f"Epoch {epoch:3d}/{tcfg['max_epochs']} | "
              f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
              f"CBF RMSE: {val_metrics['CBF_RMSE']:.3f} | "
              f"ATT RMSE: {val_metrics['ATT_RMSE']:.4f} | "
              f"LR: {current_lr:.2e} | {epoch_time:.1f}s")

        # Best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': cfg,
                'n_params': n_params,
            }, str(final_dir / 'best_model.pt'))
            print(f"  [*] New best model saved (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1

        # Latest checkpoint (for resuming)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'history': history,
            'config': cfg,
        }, str(ckpt_dir / 'latest_checkpoint.pt'))

        # Early stopping
        if patience_counter >= tcfg['early_stopping_patience']:
            print(f"\n[*] Early stopping at epoch {epoch} (patience={tcfg['early_stopping_patience']})")
            break

    total_time = time.time() - train_start
    print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save training history
    with open(str(log_dir / 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Save run summary
    summary = {
        'total_training_time_s': float(total_time),
        'total_epochs': int(len(history)),
        'best_epoch': int(min(range(len(history)), key=lambda i: history[i]['val_loss'])),
        'best_val_loss': float(best_val_loss),
        'n_params': int(n_params),
        'device': str(device),
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A',
        'N_TRAIN': int(data['N_TRAIN']),
        'N_VAL': int(data['N_VAL']),
        'N_TEST': int(data['N_TEST']),
        'seed': int(cfg['training']['seed']),
        'pytorch_version': str(torch.__version__),
        'python_version': str(sys.version),
    }
    with open(str(log_dir / 'run_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Best epoch: {summary['best_epoch']}, val_loss={best_val_loss:.6f}")
    return history, summary


def main():
    parser = argparse.ArgumentParser(description='Train PC-PLDNet')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).resolve().parent.parent / args.config
    with open(str(config_path), 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"Config: {config_path}")
    train(cfg, resume=args.resume)


if __name__ == '__main__':
    main()

