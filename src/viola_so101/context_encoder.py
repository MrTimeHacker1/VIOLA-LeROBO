"""Context feature encoders (VIOLA Sec. 3.2 + Appendix A).

Three context tokens per frame (dual-camera SO-101 => all three present):

  global        : Spatial Softmax over the workspace spatial feature map (the
                  SAME map produced by RegionEncoder's trunk) -> encodes task
                  stage.
  eye-in-hand   : a SEPARATE, from-scratch ResNet-18 over the wrist image,
                  followed by Spatial Softmax -> sees what the gripper occludes.
                  (Paper: "we use a ResNet-18 backbone followed by Spatial
                  Softmax to get eye-in-hand features.")
  proprioceptive: a single linear layer over the robot state.

The wrist trunk does NOT share weights with the workspace trunk. The wrist
pathway is built whenever ``wrist_image_key`` is set (the SO-101 default), so
pretrain and finetune use an identical token schema.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VIOLAConfig
from .region_encoder import build_spatial_trunk


class SpatialSoftmax(nn.Module):
    """Expected 2D coordinate per feature channel (Finn et al., 2016)."""

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        ys = torch.linspace(-1.0, 1.0, h, device=feat.device, dtype=feat.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=feat.device, dtype=feat.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        gx = gx.reshape(1, 1, h * w)
        gy = gy.reshape(1, 1, h * w)
        attn = F.softmax(feat.reshape(b, c, h * w), dim=-1)        # [B,C,HW]
        ex = (attn * gx).sum(-1)                                   # [B,C]
        ey = (attn * gy).sum(-1)                                   # [B,C]
        return torch.cat([ex, ey], dim=-1)                        # [B,2C]


class ContextEncoder(nn.Module):
    def __init__(self, cfg: VIOLAConfig):
        super().__init__()
        self.cfg = cfg
        self.use_wrist = cfg.wrist_image_key is not None

        self.global_ss = SpatialSoftmax()
        self.global_proj = nn.Linear(cfg.spatial_channels * 2, cfg.token_dim)

        if self.use_wrist:
            self.wrist_trunk = build_spatial_trunk()              # separate weights
            self.wrist_ss = SpatialSoftmax()
            self.wrist_proj = nn.Linear(cfg.spatial_channels * 2, cfg.token_dim)

        self.proprio_proj = nn.Linear(cfg.state_dim, cfg.token_dim)

    def forward(self, workspace_fmap, wrist_image, state):
        """workspace_fmap: [N,C,16,16]; wrist_image: [N,3,S,S] or None;
        state: [N, state_dim]. Returns context tokens [N, n_context, token_dim].

        Token order: [global, wrist(if present), proprio]."""
        tokens = [self.global_proj(self.global_ss(workspace_fmap))]
        if self.use_wrist:
            wf = self.wrist_trunk(wrist_image)
            tokens.append(self.wrist_proj(self.wrist_ss(wf)))
        tokens.append(self.proprio_proj(state))
        return torch.stack(tokens, dim=1)                         # [N,n_ctx,D]
