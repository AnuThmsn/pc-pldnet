# Physics-Consistency Loss: Forensic Audit & Validation Report

**Project:** PC-PLDNet (Physics-Consistent PLD-Conditioned Network)  
**Date:** October 2026  
**Artifact:** `diagnostics/physics_loss_validation.md`  
**Data Sources:** `diagnostics/physics_validation_data.json`, `src/physics.py`, `src/losses.py`, `src/dataset.py`  

---

## 1. Executive Summary

The physics-consistency loss was proposed as a core inductive bias in **PC-PLDNet**, intended to regularize parameter estimates by passing predicted CBF ($f$) and ATT ($\delta$) through the differentiable Buxton pCASL kinetic forward model and penalizing discrepancies with the measured 4-PLD signals.

This report provides a rigorous empirical and analytical audit of:
1. **The Observation Model Convention**: Assessing whether the assumed expectation $\mathbb{E}[X] = 2 \cdot S$ holds under the simulated dual-Rician acquisition noise.
2. **Gradient Backpropagation**: Quantifying gradient norms from CBF supervision, ATT supervision, and physics consistency across the shared representation trunk and the individual output heads.
3. **Parameter Bias & Calibration**: Evaluating whether active physics regularization genuinely improves held-out parameter estimation or introduces systematic CBF bias due to rectified Rician noise.

---

## 2. Gradient Norm & Loss Component Instrumentation

Gradient norms were instrumented on the PyTorch model across representative batches of training data, evaluating gradients propagating to:
- **Shared Representation Trunk**: `encoder` (Module A) + `conv_extractor` (Module B) + `attention` (Module C) + `shared_mlp` (Module D).
- **CBF Prediction Head**: `cbf_head` (Module E).
- **ATT Prediction Head**: `att_head` (Module E).

### Empirical Gradient Measurements

| Parameter Group / Metric | Original Objective (`SmoothL1`, $w_{\text{ATT}}=2$, $\lambda_{\text{phys}}=0.01$) | Rebalanced Objective ($L_1$, $w_{\text{ATT}}=40$, $\lambda_{\text{phys}}=0.5$) |
|---|---|---|
| **Shared Trunk: CBF Grad Norm** | **2.6072** | **3.7834** |
| **Shared Trunk: ATT Grad Norm** | **0.2113** | **17.7541** |
| **Shared Trunk: Physics Grad Norm** | **0.003591** | **0.182698** |
| **Trunk CBF-to-ATT Gradient Ratio** | **12.34 : 1** (CBF dominates) | **0.21 : 1** (Balanced parity) |
| **Physics-to-Supervised Ratio** | **0.127%** (Vanished) | **0.848%** (Active) |
| **CBF Head Grad Norm** | **3.6899** | **5.2122** |
| **ATT Head Grad Norm** | **0.1141** | **9.0328** |
| **Unweighted Loss $\mathcal{L}_{\text{CBF}}$** | 2.9650 | 3.3957 |
| **Unweighted Loss $\mathcal{L}_{\text{ATT}}$** | 0.0645 s | 0.2640 s |
| **Unweighted Loss $\mathcal{L}_{\text{physics}}$** | 0.1174 | 0.1191 |

### Key Diagnostic Findings:
1. **Vanished Physics Gradients in Original Formulation**:
   - In the original objective, the physics gradient norm arriving at the shared trunk was **$0.00359$**, representing just **$0.127\%$** of the supervised gradients ($1/780\text{th}$ of the CBF gradient).
   - This was caused by two compounding factors: dividing the residual by the per-sample mean signal ($\sim 500$) and then passing that $<0.01$ value through quadratic `SmoothL1Loss` ($\beta=1.0$), squaring the denominator ($500^2 = 250,000$).
   - *Result*: The physics loss was mathematically inert during original model training.
2. **Reactivation under Linear Relative $L_1$**:
   - Formulating the physics loss as a linear relative $L_1$ norm ($\|2S - X\| / \text{scale}$) with $\lambda_{\text{phys}} = 0.5$ increased the physics gradient norm by **$50.9\times$** (from $0.00359$ to $0.1827$).

---

## 3. Dual-Rician Observation Convention Audit

### Mathematical Analysis

In `src/dataset.py`, noisy measurements are generated according to the dual-coil magnitude sum:
$$m_c = \sqrt{(S + e_1)^2 + e_2^2}, \quad m_l = \sqrt{(S + e_3)^2 + e_4^2}$$
$$X = m_c + m_l$$
where $e_i \sim \mathcal{N}(0, \sigma^2)$ are independent Gaussian noise realizations, and $S$ is the noiseless Buxton pCASL difference signal.

Each component $m \in \{m_c, m_l\}$ follows a Rice distribution with non-centrality parameter $\nu = S$ and scale $\sigma$:
$$\mathbb{E}[m] = \sigma \sqrt{\frac{\pi}{2}} L_{1/2}\left(-\frac{S^2}{2\sigma^2}\right)$$
where $L_{1/2}(x) = e^{x/2} \left( (1-x) I_0(-x/2) - x I_1(-x/2) \right)$ is a Laguerre polynomial.

Because $m_c$ and $m_l$ are independently identically distributed:
$$\mathbb{E}[X] = \mathbb{E}[m_c] + \mathbb{E}[m_l] = 2 \sigma \sqrt{\frac{\pi}{2}} L_{1/2}\left(-\frac{S^2}{2\sigma^2}\right)$$

### Asymptotic Limits:
1. **High SNR ($\frac{S}{\sigma} \gg 1$)**:
   $$\mathbb{E}[m] \approx S + \frac{\sigma^2}{2S} + \mathcal{O}(S^{-3}) \implies \mathbb{E}[X] \approx 2S + \frac{\sigma^2}{S} \approx 2S$$
2. **Low SNR ($\frac{S}{\sigma} \to 0$)**:
   $$\mathbb{E}[m] \to \sigma \sqrt{\frac{\pi}{2}} \approx 1.2533 \sigma > 0$$
   $$\mathbb{E}[X] \to 2.5066 \sigma > 0$$

When the true signal $S$ is small (e.g. at late PLDs or in hypoperfused tissue), **the expectation of the observation does NOT vanish**. It hits the **Rician noise floor** of $+2.5066 \sigma$.

### Empirical Verification on 5,000 Test Samples

The table below reports empirical measurements of clean signal $S$, assumed observation $2S$, actual noisy observation $X = m_c + m_l$, and the resulting Rician excess bias:

| SNR Stratum | N Samples | Mean Clean $S$ | Assumed Obs ($2S$) | Actual Obs ($X$) | Rician Excess Bias ($X - 2S$) | Relative Bias |
|---|---|---|---|---|---|---|
| **$\text{SNR} < 7.5$ (Low SNR)** | 2,481 (49.6%) | 152.0 | 304.0 | 331.2 | **+27.2** | **+8.96%** |
| **$[7.5, 12.5)$ (~SNR 10)** | 977 (19.5%) | 327.6 | 655.3 | 662.0 | **+6.7** | **+1.02%** |
| **$[12.5, 17.5)$ (~SNR 15)** | 422 (8.4%) | 327.1 | 654.1 | 655.7 | **+1.6** | **+0.25%** |
| **$[17.5, 25.0)$ (~SNR 20)** | 300 (6.0%) | 320.2 | 640.4 | 641.7 | **+1.3** | **+0.20%** |
| **$\text{SNR} \ge 25.0$ (High SNR)** | 597 (11.9%) | 334.0 | 668.1 | 668.3 | **+0.2** | **+0.03%** |

---

## 4. Why Naive Physics Regularization Induces Positive CBF Bias

1. **Noise-Induced Asymmetry**:
   - In `src/physics.py`: `expected_observation(cbf, att) = 2.0 * forward(cbf, att) = 2S`.
   - The physics loss penalizes:
     $$\mathcal{L}_{\text{phys}} = \frac{\| 2 \cdot S(\hat{f}, \hat{\delta}) - X \|}{\text{scale}}$$
   - Because $49.2\%$ of all test samples reside in the low-SNR regime ($\text{SNR} < 7.5$), the empirical observations $X$ are systematically **$+8.96\%$ higher than $2S$** due to Rician noise rectification.
2. **Upward Pressure on CBF**:
   - Because $S(f, \delta) \propto f$ (CBF), minimizing the discrepancy with the elevated noisy observations $X$ exerts a persistent gradient pressure pushing $\hat{f}$ upward.
   - In the original model, this effect was suppressed because the physics loss gradient norm was tiny ($0.00359$), resulting in a near-zero overall CBF bias of **$+0.0422\,\text{ml/100g/min}$**.
   - When the physics loss was reactivated with linear relative $L_1$ ($\lambda = 0.5$, gradient norm $0.1827$), the overall CBF bias **increased five-fold to $+0.2124\,\text{ml/100g/min}$**.

---

## 5. Conclusions & Recommendations

1. **The Observation Model is Flawed at Low SNR**:
   - The assumption $\mathbb{E}[X] = 2S$ is valid only for $\text{SNR} \ge 15$. At lower SNRs, rectified Rician noise introduces a positive offset up to $+9\%$.
2. **Physics Regularization Does Not Drive the ATT Improvement**:
   - The major ATT improvements observed in PC-PLDNet (MAE dropping from $0.2474\,\text{s}$ to $0.2337\,\text{s}$ overall, and $0.1884\,\text{s}$ to $0.1534\,\text{s}$ in the identifiable regime) are **driven entirely by supervised loss rebalancing** ($w_{\text{ATT}}=40.0$ and $L_1$ regression).
   - The physics loss, when active, contributes no observable timing benefit and penalizes CBF calibration by injecting Rician noise bias.
3. **Recommendation**:
   - In standard 4-PLD single-voxel settings without explicit Rician expectation correction, the physics loss should either:
     a) Be deactivated or given negligible weight ($\lambda_{\text{phys}} \le 0.01$), relying on supervised $L_1$ rebalancing; or
     b) Incorporate the analytical Rician mean correction $\mathbb{E}[X] = 2 \sigma \sqrt{\pi/2} L_{1/2}(-S^2/(2\sigma^2))$ when noise level $\sigma$ is estimated.

