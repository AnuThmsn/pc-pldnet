# PC-PLDNet: Physics-Consistent PLD-Conditioned Network for ASL Parameter Estimation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation, training pipeline, evaluation scripts, and research artifacts for **PC-PLDNet (Physics-Consistent PLD-Conditioned Network)**, designed to estimate Cerebral Blood Flow (CBF, $\text{mL}/100\,\text{g}/\text{min}$) and Arterial Transit Time (ATT, $\text{s}$) from 4-delay multi-PLD ASL MRI ($t \in [1.525, 2.025, 2.525, 3.025]\,\text{s}$).

## Architecture Overview

PC-PLDNet embeds physical delay conditioning, cross-delay attention, bounded parameter heads, and self-supervised kinetic reconstruction into a single unified model ($55,490$ parameters):

```
       Input (B, 4) Normalized ASL Measurements
                         │
        ┌────────────────┴────────────────┐
        │  Module A: PLD-Conditioned Enc  │
        │  (signal proj + sinusoidal PLD) │ ──> (B, 4, 48) tokens
        └────────────────┬────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │  Module B: 1D Conv Extractor   │
        │  (Conv1D-BN-GELU x 2)          │ ──> (B, 4, 64)
        └────────────────┬────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │  Module C: Cross-PLD Attention  │
        │  (Multi-head Self-Attn + Pool)  │ ──> (B, 64) fused vector
        └────────────────┬────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │  Module D: Shared MLP           │
        │  (Linear 64→128→64 + Dropout)   │ ──> (B, 64) shared latent
        └────────┬───────────────┬────────┘
                 │               │
        ┌────────┴──────┐ ┌──────┴──────┐
        │  Module E:    │ │  Module E:  │
        │  CBF Head     │ │  ATT Head   │
        │  Linear+Sig   │ │  Linear+Sig │
        └───────┬───────┘ └──────┬──────┘
                │                │
            CBF (B,)          ATT (B,)
       (0 - 100 mL/100g/min)  (0.5 - 3.0 s)
                │                │
        ┌───────┴────────────────┴───────┐
        │  Module F: Physics Loss        │
        │  Differentiable Buxton Forward │ ──> L_physics
        └────────────────────────────────┘
```

## Key Empirical Findings

Evaluated on 50,000 frozen held-out test samples against the official 9-layer MLP baseline (DNN-G):
- **39.7% Parameter Reduction:** $55,490$ parameters vs. $91,952$ parameters in the baseline.
- **85% Training Data Reduction:** Trained on $300,000$ samples vs. $2,000,000$ in the baseline.
- **CBF Accuracy:** $\text{RMSE} = 4.608\,\text{mL}/100\,\text{g}/\text{min}$ ($\text{MAE} = 3.338$, $\text{CCC} = 0.987$) — parity with baseline within $\sim 0.05\,\text{mL}$.
- **ATT Accuracy:** $\text{RMSE} = 0.3590\,\text{s}$ ($\text{MAE} = 0.2474$, $\text{CCC} = 0.856$) — outperforms baseline ($0.3622\,\text{s}$).
- **Lower Systematic Bias:** 53.5% lower CBF bias ($+0.0422$ vs $+0.0907$) and 75.2% lower ATT bias ($-0.0072\,\text{s}$ vs $+0.0291\,\text{s}$).
- **High-Throughput Inference:** $1.254\,\text{s}$ for 50,000 samples ($0.025\,\text{ms}$ per voxel).

## Repository Structure

```text
├── configs/
│   └── default.yaml              # Single source of truth configuration
├── src/
│   ├── model.py                  # PC-PLDNet architecture (Modules A-E)
│   ├── physics.py                # Differentiable ASL Buxton forward model
│   ├── dataset.py                # Synthetic data generator & PyTorch Datasets
│   ├── losses.py                 # Multi-task regression + physics loss
│   ├── train.py                  # Training pipeline with early stopping & checkpointing
│   ├── evaluate.py               # Comprehensive evaluation & figure generation
│   └── metrics.py                # RMSE, MAE, Bias, R², Pearson r, CCC
├── data/
│   └── README.md                 # Signal model and noise generation specification
├── models/
│   ├── checkpoints/              # Resumable training checkpoints
│   └── final/                    # Best model weights and normalization stats
├── experiments/
│   ├── logs/                     # Epoch logs and run summaries
│   ├── predictions/              # Saved numpy test set predictions
│   ├── metrics/                  # Final overall and SNR-stratified metrics (CSV/JSON)
│   └── figures/                  # Publication-quality diagnostic plots
├── tests/
│   ├── test_model.py             # Unit tests for architecture & gradients
│   ├── test_physics.py           # Unit tests for forward model & differentiability
│   └── test_data.py              # Unit tests for data generation & metrics
├── docs/
│   ├── implementation.md         # In-depth architectural & implementation audit
│   ├── evaluation_protocol.md    # Formal testing protocol & baseline comparison rules
│   └── final_report.md           # Full academic report of experimental results
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Installation
```bash
git clone https://github.com/AnuThmsn/pc-pldnet.git
cd pc-pldnet
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
```bash
pytest tests -v
```

### 3. Train PC-PLDNet
```bash
python src/train.py --config configs/default.yaml
```

To resume an interrupted run:
```bash
python src/train.py --config configs/default.yaml --resume
```

### 4. Evaluate Held-Out Test Set
```bash
python src/evaluate.py --config configs/default.yaml
```

## Citation & Documentation
For full experimental methodology, biophysical constants, and comparison tables, see [docs/final_report.md](docs/final_report.md).
