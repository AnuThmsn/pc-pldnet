# Synthetic ASL Data Generation and Storage

## Signal Simulation Overview

PC-PLDNet is trained and evaluated using the exact Buxton-style pCASL kinetic signal model and dual-Rician noise generation implemented in the parent repository (`data_gen.py`).

### Kinetic Model Equations

For labeling duration $\tau = 1.8\text{ s}$, arterial blood relaxation $T_{1a} = 1.66\text{ s}$, tissue relaxation $T_{1t} = 1.2\text{ s}$, labeling efficiency $\alpha = 0.85$, background suppression factor $\beta = 0.75$, and blood-brain partition coefficient $\lambda = 0.90\text{ mL/g}$:

$$\Delta M(t) = 2 \alpha \beta T_{1t} \frac{1}{\lambda} f e^{-\delta / T_{1a}} \left( e^{-\max(t - \delta, 0)/T_{1t}} - e^{-\max(\tau + t - \delta, 0)/T_{1t}} \right)$$

where:
- $f = \frac{\text{CBF}}{6000 \cdot \lambda}$ is perfusion rate per second.
- $\delta = \text{ATT}$ is arterial transit time in seconds.
- $t \in \{1.525, 2.025, 2.525, 3.025\}\text{ s}$ are the 4 post-labeling delays.
- Raw signal is scaled by $\text{SCALE} = 100,000$.

### Noise and Observation Model

Synthetic observations simulate dual-Rician magnitude detection across separate control and label channels:
$$M_c = \sqrt{(S + e_1)^2 + e_2^2}, \quad M_l = \sqrt{(S + e_3)^2 + e_4^2}$$
$$X = M_c + M_l$$
where $e_i \sim \mathcal{N}(0, \sigma^2)$, with $\sigma \in [0, \sigma_{\max}]$ sampled uniformly across discrete noise levels ($\sigma_{\max} = \text{SIG\_REF\_SC} / 5.0 \approx 66.84$).

### Dataset Splits and Reproducibility

- **Training:** 300,000 samples (seed: 42, 100 noise levels)
- **Validation:** 50,000 samples (seed: 99, 100 noise levels)
- **Test:** 50,000 samples (seed: 7, 51 noise levels)
- **Leakage Prevention:** Normalization statistics ($\mu_X, \sigma_X$) are fitted strictly on the training set and applied to validation and test sets.

