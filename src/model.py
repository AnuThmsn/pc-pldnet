"""
model.py — Physics-Consistent PLD-Conditioned Network (PC-PLDNet).

Architecture:
  Module A: PLD-conditioned encoding — (signal, PLD) → token
  Module B: 1D Conv feature extraction over 4 tokens
  Module C: Multi-head cross-PLD attention → fused representation
  Module D: Shared MLP for joint CBF/ATT features
  Module E: Separate CBF and ATT output heads with bounded outputs
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PLDConditionedEncoder(nn.Module):
    """Module A: Encode each (signal_value, PLD_value) pair into a token.

    Uses sinusoidal positional encoding for PLD timing plus a learned
    signal projection. Output: one token per PLD measurement.
    """

    def __init__(self, pld_embed_dim=16, signal_embed_dim=32):
        super().__init__()
        self.pld_embed_dim = pld_embed_dim
        self.signal_embed_dim = signal_embed_dim
        self.token_dim = pld_embed_dim + signal_embed_dim

        # Learned signal projection: scalar → signal_embed_dim
        self.signal_proj = nn.Sequential(
            nn.Linear(1, signal_embed_dim),
            nn.GELU(),
            nn.Linear(signal_embed_dim, signal_embed_dim),
        )

        # PLD encoding: sinusoidal (fixed) + learned linear projection
        self.pld_proj = nn.Linear(pld_embed_dim, pld_embed_dim)

    def _sinusoidal_encoding(self, pld_values, dim):
        """Sinusoidal positional encoding for PLD timing values.

        Args:
            pld_values: (B, n_plds) or (n_plds,) PLD values in seconds
            dim: embedding dimension (must be even)

        Returns:
            encoding: (..., dim)
        """
        if pld_values.dim() == 1:
            pld_values = pld_values.unsqueeze(0)

        half_dim = dim // 2
        # Frequency bands spanning a useful range for PLD values (0.5 - 5s)
        freqs = torch.exp(
            torch.arange(half_dim, device=pld_values.device, dtype=torch.float32)
            * -(math.log(10.0) / half_dim)
        )  # (half_dim,)

        # (B, n_plds, 1) * (1, 1, half_dim) → (B, n_plds, half_dim)
        angles = pld_values.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return encoding  # (B, n_plds, dim)

    def forward(self, signals, pld_values):
        """
        Args:
            signals: (B, n_plds) normalized signal values
            pld_values: (n_plds,) PLD values in seconds (registered buffer)

        Returns:
            tokens: (B, n_plds, token_dim)
        """
        B, n_plds = signals.shape

        # Signal encoding: (B, n_plds) → (B, n_plds, signal_embed_dim)
        sig_features = self.signal_proj(signals.unsqueeze(-1))  # (B, n_plds, 1) → (B, n_plds, SE)

        # PLD encoding: sinusoidal + projection
        pld_enc = self._sinusoidal_encoding(pld_values, self.pld_embed_dim)  # (1, n_plds, PE)
        pld_enc = pld_enc.expand(B, -1, -1)  # (B, n_plds, PE)
        pld_features = self.pld_proj(pld_enc)  # (B, n_plds, PE)

        # Concatenate: (B, n_plds, token_dim)
        tokens = torch.cat([sig_features, pld_features], dim=-1)
        return tokens


class ConvFeatureExtractor(nn.Module):
    """Module B: 1D convolutional feature extraction over PLD tokens.

    Applies 1D convolutions along the PLD dimension to capture
    local signal decay patterns.
    """

    def __init__(self, in_channels, conv_channels, kernel_size=3):
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out in conv_channels:
            layers.extend([
                nn.Conv1d(ch_in, ch_out, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(ch_out),
                nn.GELU(),
            ])
            ch_in = ch_out
        self.conv = nn.Sequential(*layers)

    def forward(self, tokens):
        """
        Args:
            tokens: (B, n_plds, token_dim)

        Returns:
            features: (B, n_plds, conv_channels[-1])
        """
        # Conv1d expects (B, C, L), so transpose
        x = tokens.transpose(1, 2)  # (B, token_dim, n_plds)
        x = self.conv(x)            # (B, conv_channels[-1], n_plds)
        return x.transpose(1, 2)    # (B, n_plds, conv_channels[-1])


class CrossPLDAttention(nn.Module):
    """Module C: Multi-head attention across PLD tokens → fused vector."""

    def __init__(self, embed_dim, num_heads=4, dropout=0.05):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Learnable query for pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(self, tokens):
        """
        Args:
            tokens: (B, n_plds, embed_dim)

        Returns:
            fused: (B, embed_dim) — single fused representation
        """
        B = tokens.shape[0]

        # Self-attention among PLD tokens (with residual)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)

        # Pool using learnable query
        query = self.pool_query.expand(B, -1, -1)  # (B, 1, embed_dim)
        pooled, _ = self.attn(query, tokens, tokens)
        fused = pooled.squeeze(1)  # (B, embed_dim)

        return fused


class SharedMLP(nn.Module):
    """Module D: Shared representation MLP."""

    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_d, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_d = h
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x):
        return self.mlp(x)


class OutputHead(nn.Module):
    """Module E: Separate prediction head for CBF or ATT.

    Uses sigmoid output scaled to physical range for bounded predictions.
    """

    def __init__(self, input_dim, hidden_dims, y_min, y_max):
        super().__init__()
        self.y_min = y_min
        self.y_max = y_max
        self.y_range = y_max - y_min

        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_d, h),
                nn.GELU(),
            ])
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, input_dim) shared features

        Returns:
            pred: (B,) prediction in physical units, bounded to [y_min, y_max]
        """
        raw = self.head(x).squeeze(-1)  # (B,)
        # Sigmoid scaling to physical range
        pred = self.y_min + self.y_range * torch.sigmoid(raw)
        return pred


class PCPLDNet(nn.Module):
    """Physics-Consistent PLD-Conditioned Network (PC-PLDNet).

    Full architecture combining Modules A-E.
    """

    def __init__(self, cfg):
        super().__init__()
        mcfg = cfg['model']
        tcfg = cfg['targets']
        plds = cfg['physics']['plds']

        # Register PLD values as buffer
        self.register_buffer('pld_values', torch.tensor(plds, dtype=torch.float32))
        self.n_plds = len(plds)

        # Module A: PLD-conditioned encoder
        self.encoder = PLDConditionedEncoder(
            pld_embed_dim=mcfg['pld_embed_dim'],
            signal_embed_dim=mcfg['signal_embed_dim'],
        )
        token_dim = self.encoder.token_dim

        # Module B: 1D Conv feature extractor
        self.conv_extractor = ConvFeatureExtractor(
            in_channels=token_dim,
            conv_channels=mcfg['conv_channels'],
            kernel_size=mcfg['conv_kernel_size'],
        )
        conv_out_dim = mcfg['conv_channels'][-1]

        # Project to attention dimension if needed
        attn_dim = mcfg['attn_dim']
        self.attn_proj = nn.Linear(conv_out_dim, attn_dim) if conv_out_dim != attn_dim else nn.Identity()

        # Module C: Cross-PLD attention
        self.attention = CrossPLDAttention(
            embed_dim=attn_dim,
            num_heads=mcfg['attn_heads'],
            dropout=mcfg['attn_dropout'],
        )

        # Module D: Shared MLP
        self.shared_mlp = SharedMLP(
            input_dim=attn_dim,
            hidden_dims=mcfg['shared_hidden'],
            dropout=mcfg['shared_dropout'],
        )
        shared_out = self.shared_mlp.output_dim

        # Module E: Output heads
        self.cbf_head = OutputHead(
            input_dim=shared_out,
            hidden_dims=mcfg['cbf_head_hidden'],
            y_min=tcfg['CBF_MIN'],
            y_max=tcfg['CBF_MAX'],
        )
        self.att_head = OutputHead(
            input_dim=shared_out,
            hidden_dims=mcfg['att_head_hidden'],
            y_min=tcfg['ATT_MIN'],
            y_max=tcfg['ATT_MAX'],
        )

    def forward(self, x):
        """
        Args:
            x: (B, 4) normalized signal measurements

        Returns:
            cbf_pred: (B,) CBF in ml/100g/min
            att_pred: (B,) ATT in seconds
        """
        # Module A: PLD-conditioned encoding
        tokens = self.encoder(x, self.pld_values)  # (B, 4, token_dim)

        # Module B: 1D conv features
        tokens = self.conv_extractor(tokens)         # (B, 4, conv_out)

        # Project to attention dim
        tokens = self.attn_proj(tokens)              # (B, 4, attn_dim)

        # Module C: Cross-PLD attention
        fused = self.attention(tokens)               # (B, attn_dim)

        # Module D: Shared MLP
        shared = self.shared_mlp(fused)              # (B, shared_out)

        # Module E: Separate heads
        cbf_pred = self.cbf_head(shared)             # (B,)
        att_pred = self.att_head(shared)              # (B,)

        return cbf_pred, att_pred

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg, device='cpu'):
    """Build PC-PLDNet model and move to device."""
    model = PCPLDNet(cfg).to(device)
    return model

