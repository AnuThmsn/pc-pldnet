"""
losses.py — Loss functions for PC-PLDNet training.

L_total = L_CBF + w_ATT * L_ATT + lambda_phys * L_physics

- L_CBF, L_ATT: SmoothL1 regression losses in physical units
- L_physics: Physics-consistency reconstruction loss comparing
  forward-modeled signals from predicted parameters against observations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConsistencyLoss(nn.Module):
    """Module F: ASL signal reconstruction loss.

    Given predicted CBF and ATT, reconstruct expected observations via
    the differentiable forward model and compare against measured signals.
    """

    def __init__(self, forward_model, X_mean, X_std, loss_type='L1'):
        """
        Args:
            forward_model: ASLForwardModel instance
            X_mean: (1, 4) training mean for denormalization
            X_std: (1, 4) training std for denormalization
            loss_type: 'L1' (linear, preserves gradient) or 'SmoothL1'
        """
        super().__init__()
        self.forward_model = forward_model
        self.loss_type = loss_type
        self.register_buffer('X_mean', torch.as_tensor(X_mean, dtype=torch.float32))
        self.register_buffer('X_std', torch.as_tensor(X_std, dtype=torch.float32))

    def forward(self, cbf_pred, att_pred, x_normalized):
        """
        Args:
            cbf_pred: (B,) predicted CBF in ml/100g/min
            att_pred: (B,) predicted ATT in seconds
            x_normalized: (B, 4) normalized input signals

        Returns:
            loss: scalar physics reconstruction loss
        """
        # Denormalize inputs to raw observation scale
        x_raw = x_normalized * self.X_std + self.X_mean  # (B, 4)

        # Reconstruct expected observations from predictions
        # expected_obs = 2 * clean_signal (since X = mc + ml ≈ 2*S for reasonable SNR)
        x_reconstructed = self.forward_model.expected_observation(cbf_pred, att_pred)  # (B, 4)

        # Normalized reconstruction loss
        scale = x_raw.abs().mean(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        if self.loss_type == 'SmoothL1':
            loss = F.smooth_l1_loss(x_reconstructed / scale, x_raw / scale)
        else:
            loss = F.l1_loss(x_reconstructed / scale, x_raw / scale)

        return loss


class PCPLDNetLoss(nn.Module):
    """Combined training loss for PC-PLDNet.

    L_total = L_CBF + w_ATT * L_ATT + lambda_phys * L_physics
    """

    def __init__(self, forward_model, X_mean, X_std, cfg):
        super().__init__()
        tcfg = cfg['training']

        self.att_weight = tcfg['att_loss_weight']
        self.physics_lambda = tcfg['physics_lambda']

        reg_type = tcfg.get('regression_loss', 'SmoothL1')
        if reg_type in ('L1', 'MAE'):
            self.cbf_loss_fn = nn.L1Loss()
            self.att_loss_fn = nn.L1Loss()
        else:
            beta = tcfg.get('smoothl1_beta', 1.0)
            self.cbf_loss_fn = nn.SmoothL1Loss(beta=beta)
            self.att_loss_fn = nn.SmoothL1Loss(beta=beta)

        phys_loss_type = tcfg.get('physics_loss_type', 'L1')
        self.physics_loss = PhysicsConsistencyLoss(forward_model, X_mean, X_std, loss_type=phys_loss_type)

    def forward(self, cbf_pred, att_pred, cbf_true, att_true, x_normalized):
        """
        Args:
            cbf_pred, att_pred: (B,) predictions in physical units
            cbf_true, att_true: (B,) ground truth in physical units
            x_normalized: (B, 4) normalized inputs

        Returns:
            total_loss: scalar
            loss_dict: dict of individual loss components
        """
        l_cbf = self.cbf_loss_fn(cbf_pred, cbf_true)
        l_att = self.att_loss_fn(att_pred, att_true)
        l_phys = self.physics_loss(cbf_pred, att_pred, x_normalized)

        total = l_cbf + self.att_weight * l_att + self.physics_lambda * l_phys

        return total, {
            'L_CBF': l_cbf.item(),
            'L_ATT': l_att.item(),
            'L_physics': l_phys.item(),
            'L_total': total.item(),
        }

