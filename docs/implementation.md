# PC-PLDNet: Technical Implementation Audit & Architecture Specification

## 1. Audit of Existing Codebase & Environment

### 1.1 Validated Signal Generation & Physics Parameters
From `data_gen.py` and `ddn_asl.ipynb`:
- **PLD Schedule:** $t = [1.525, 2.025, 2.525, 3.025]\text{ s}$ (4 post-labeling delays).
- **Labeling Duration ($\tau$):** $1.8\text{ s}$.
- **Tissue Relaxation ($T_{1t}$):** $1.2\text{ s}$.
- **Arterial Blood Relaxation ($T_{1a}$):** $1.66\text{ s}$.
- **Labeling Efficiency ($\alpha$):** $0.85$.
- **Background Suppression ($\beta$):** $0.75$.
- **Partition Coefficient ($\lambda$):** $0.90\text{ mL/g}$.
- **Scale Factor ($\text{SCALE}$):** $100,000.0$.
- **Reference Signal ($\text{SIG\_REF\_SC}$):** $334.2039$.
- **Noise Upper Bound ($\sigma_{\max}$):** $\text{SIG\_REF\_SC} / 5.0 = 66.84078$.
- **Observation Model:** Dual-Rician magnitude sum $X = M_c + M_l = \sqrt{(S+e_1)^2 + e_2^2} + \sqrt{(S+e_3)^2 + e_4^2}$, with $E[X] \approx 2S$ for high SNR.

### 1.2 Existing Baseline Architecture (DNN-G / StandardizedNet)
- **CBF Network:** 9 hidden layers $\times$ 50 neurons with ELU activations, Kaiming-normal initialization, predicting standardized CBF ($18,651$ parameters). Checkpoint: `asl_dnn_papermatched/CBF_papermatched_9x50_MAE_best.pth`.
- **ATT Network:** 9 hidden layers $\times$ 100 neurons with ELU activations, Kaiming-normal initialization, predicting standardized ATT ($73,301$ parameters). Checkpoint: `asl_dnn_papermatched/ATT_papermatched_9x100_MAE_best.pth`.
- **Combined Baseline Parameters:** $18,651 + 73,301 = 91,952$ parameters across two independent networks.
- **Baseline Training:** Adam optimizer ($\text{lr} = 10^{-3}$), L1 loss (MAE), batch size 512, gradient clipping 1.0, 2,000,000 training samples.

### 1.3 Normalization & Target Ranges
- **Input Normalization:** Fitted strictly on training set:
  $$\mu_X \approx [662.41, 566.48, 442.60, 317.14], \quad \sigma_X \approx [476.38, 341.60, 250.00, 176.14]$$
- **Target Ranges:**
  - $\text{CBF} \in [0.0, 100.0]\text{ mL}/100\,\text{g}/\text{min}$ ($\mu_{\text{CBF}} \approx 49.99, \sigma_{\text{CBF}} \approx 28.86$)
  - $\text{ATT} \in [0.5, 3.0]\text{ s}$ ($\mu_{\text{ATT}} \approx 1.75, \sigma_{\text{ATT}} \approx 0.72$)

### 1.4 Compute Environment & Hardware
- **OS:** Windows 11 (64-bit)
- **Python Version:** 3.10.11
- **PyTorch Version:** 2.5.1+cu121 (CUDA enabled)
- **GPU:** NVIDIA GeForce GTX 1650 (4.29 GB VRAM)
- **System RAM:** 16.0 GB total (~1.8 GB free at start of task)

---

## 2. PC-PLDNet Architecture Specification

PC-PLDNet unifies CBF and ATT estimation into a single, compact, physics-consistent architecture ($53,890$ parameters):

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

### Module Descriptions
- **Module A (PLD-Conditioned Encoder):** Encodes each measurement $(x_i, t_i)$ into a token using a learned 2-layer signal MLP ($1 \to 32 \to 32$) concatenated with a 16-dim sinusoidal timing encoding of $t_i$, yielding a 48-dim token per delay.
- **Module B (1D Convolutional Extractor):** Operates across the sequence of 4 delay tokens with kernel size 3 and padding 1, expanding channels from 48 to 64 to model local decay trends.
- **Module C (Cross-PLD Multi-Head Attention):** Uses 4-head self-attention with residual normalization, followed by cross-attention pooling with a learnable query to condense the 4 tokens into a single 64-dim representation.
- **Module D (Shared Latent MLP):** A 2-layer feed-forward network ($64 \to 128 \to 64$) with GELU activations and 0.1 dropout for joint parameter representation.
- **Module E (Bounded Output Heads):** Separate 2-layer regression heads ($64 \to 32 \to 1$) followed by scaled sigmoid transformations:
  $$\hat{\text{CBF}} = \text{CBF}_{\min} + (\text{CBF}_{\max} - \text{CBF}_{\min}) \cdot \sigma(\mathbf{z}_{\text{CBF}})$$
  $$\hat{\text{ATT}} = \text{ATT}_{\min} + (\text{ATT}_{\max} - \text{ATT}_{\min}) \cdot \sigma(\mathbf{z}_{\text{ATT}})$$
  This guarantees physically valid outputs without post-hoc heuristic clipping.
- **Module F (Physics-Consistency Loss):** A fully differentiable Buxton kinetic forward model generates the expected ASL difference signals from $(\hat{\text{CBF}}, \hat{\text{ATT}})$, comparing them against measured inputs using normalized Smooth L1 loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CBF}} + w_{\text{ATT}} \mathcal{L}_{\text{ATT}} + \lambda_{\text{phys}} \mathcal{L}_{\text{physics}}$$
  with $w_{\text{ATT}} = 2.0$ and $\lambda_{\text{phys}} = 0.01$.

