# PC-PLDNet: Evaluation Protocol & Baseline Comparison Rules

## 1. Evaluation Protocol

To ensure rigorous, reproducible, and leakage-free evaluation:

1. **Frozen Held-Out Test Set:** The test set ($N = 50,000$ samples) is generated using a fixed random seed (`seed=7`) and discrete noise levels. It is kept completely untouched during model training and checkpoint selection.
2. **Deterministic Checkpoint Selection:** The model checkpoint evaluated on the test set is selected solely based on minimum validation loss (`val_loss`) during training. No test-set feedback is used for hyperparameter or checkpoint tuning.
3. **Physical Units:** All metrics are reported in physical physiological units:
   - CBF: $\text{mL}/100\,\text{g}/\text{min}$
   - ATT: seconds ($\text{s}$)
4. **Primary Metrics:**
   - **RMSE (Root Mean Square Error):** $\sqrt{\frac{1}{N}\sum (\hat{y}_i - y_i)^2}$
   - **MAE (Mean Absolute Error):** $\frac{1}{N}\sum |\hat{y}_i - y_i|$
   - **Mean Bias:** $\frac{1}{N}\sum (\hat{y}_i - y_i)$
   - **Coefficient of Determination ($R^2$):** $1 - \frac{\sum (\hat{y}_i - y_i)^2}{\sum (y_i - \bar{y})^2}$
   - **Pearson Correlation ($r$):** Standard linear correlation coefficient
   - **Lin's Concordance Correlation Coefficient (CCC):** $\frac{2 \rho \sigma_x \sigma_y}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}$
5. **Inference Latency:** Measured as total wall-clock time and per-sample latency across the full 50,000-sample test batch.

---

## 2. Baseline Comparison Rules

The primary reference baseline is the repository's official **DNN-G (StandardizedNet)** model:
- Saved checkpoint: `asl_dnn_papermatched/CBF_papermatched_9x50_MAE_best.pth` and `ATT_papermatched_9x100_MAE_best.pth`.
- Architecture: 9 layers $\times$ 50 neurons for CBF, 9 layers $\times$ 100 neurons for ATT ($91,952$ total parameters).
- Training set: 2,000,000 samples.

### Criteria for Valid Comparison
Any comparison between PC-PLDNet and DNN-G must strictly adhere to:
1. **Identical Input Signals:** Both models must be evaluated on the exact same held-out test samples and noise levels.
2. **Identical Forward Physics:** Both models operate on the 4-delay schedule $[1.525, 2.025, 2.525, 3.025]\text{ s}$ generated with identical kinetic parameters.
3. **Parameter Efficiency:** The comparison must account for model capacity (PC-PLDNet: $53,890$ params vs. DNN-G: $91,952$ params).
4. **No Unjustified Superiority Claims:** Results must reflect genuine empirical differences without cherry-picking or selective reporting.

