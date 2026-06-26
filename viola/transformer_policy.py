"""Transformer policy backbone (VIOLA Sec. 3.3 + Appendix A).

Encoder-only Transformer (4 layers, 6 heads, FFN 1024, post-norm as in
Vaswani et al.). A single learnable "action token", initialized from N(0,1), is
prepended to the object-centric token sequence; its output latent is the query
that has attended over all region/context tokens and is fed to the GMM head.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from .config import VIOLAConfig


class TransformerPolicy(nn.Module):
    def __init__(self, cfg: VIOLAConfig):
        super().__init__()
        self.cfg = cfg
        self.action_token = nn.Parameter(torch.randn(1, 1, cfg.token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.token_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,          # post-norm: Y = FFN(LayerNorm(MHSA(X)))
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, n_obs_tokens, token_dim]. Returns action latent [B, token_dim]."""
        b = tokens.shape[0]
        act = self.action_token.expand(b, -1, -1)                 # [B,1,D]
        seq = torch.cat([act, tokens], dim=1)                     # [B,1+n,D]
        out = self.encoder(seq)                                   # [B,1+n,D]
        return out[:, 0]                                          # action-token output
