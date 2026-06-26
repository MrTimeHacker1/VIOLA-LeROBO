"""Full VIOLA policy (VIOLA Sec. 3).

Pipeline per training step:
  1. For each of the H+1 frames, RegionEncoder produces K region tokens and the
     workspace spatial map; ContextEncoder produces the (global, wrist, proprio)
     context tokens. Together these are the per-step feature h_t.
  2. Temporal composition: add the temporal positional encoding of each frame's
     position {0..H} to every token of that frame, then concatenate the H+1
     per-step features into the object-centric representation z_t.
  3. TransformerPolicy reasons over z_t with a prepended action token.
  4. GMMHead maps the action-token latent to a 5-mode Gaussian mixture; loss is
     the NLL of the demonstrated action.

Frame ordering convention: index 0 is the oldest frame (t-H), index H is the
current frame (t). The demonstrated target is the action at the current frame.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from .config import VIOLAConfig
from .region_encoder import RegionEncoder
from .context_encoder import ContextEncoder
from .transformer_policy import TransformerPolicy
from .gmm_head import GMMHead
from .positional_encoding import TemporalPositionalEncoding


class VIOLAPolicy(nn.Module):
    def __init__(self, cfg: VIOLAConfig):
        super().__init__()
        self.cfg = cfg
        self.region_encoder = RegionEncoder(cfg)
        self.context_encoder = ContextEncoder(cfg)
        self.temporal_pe = TemporalPositionalEncoding(
            cfg.token_dim, n_positions=cfg.n_frames, base=cfg.pe_base
        )
        self.transformer = TransformerPolicy(cfg)
        self.head = GMMHead(cfg)

    # ---- core: observations -> action-token latent -------------------------
    def encode(self, batch: dict) -> torch.Tensor:
        cfg = self.cfg
        ws = batch["workspace_images"]            # [B,T,3,S,S]
        state = batch["states"]                   # [B,T,state_dim]
        boxes = batch["proposals"]                # [B,T,K,4]
        b, t = ws.shape[:2]

        # fold time into batch so the CNNs see [B*T, ...]
        ws_flat = ws.reshape(b * t, *ws.shape[2:])
        boxes_flat = boxes.reshape(b * t, *boxes.shape[2:])
        state_flat = state.reshape(b * t, state.shape[-1])

        region_tokens, fmap = self.region_encoder(ws_flat, boxes_flat)

        wrist_flat = None
        if cfg.wrist_image_key is not None:
            wr = batch["wrist_images"]            # [B,T,3,Sw,Sw]
            wrist_flat = wr.reshape(b * t, *wr.shape[2:])
        context_tokens = self.context_encoder(fmap, wrist_flat, state_flat)

        per_step = torch.cat([region_tokens, context_tokens], dim=1)  # [B*T,P,D]
        per_step = per_step.reshape(b, t, per_step.shape[1], cfg.token_dim)

        # temporal positional encoding: add PE[i] to every token of frame i
        tpe = self.temporal_pe.table[:t]                              # [T,D]
        per_step = per_step + tpe.view(1, t, 1, cfg.token_dim)

        z = per_step.reshape(b, t * per_step.shape[2], cfg.token_dim)  # [B,T*P,D]
        return self.transformer(z)                                    # [B,D]

    # ---- training ----------------------------------------------------------
    def forward(self, batch: dict) -> torch.Tensor:
        latent = self.encode(batch)
        return self.head.loss(latent, batch["action"])

    # ---- inference ---------------------------------------------------------
    @torch.no_grad()
    def select_action(self, batch: dict, sample: bool = False) -> torch.Tensor:
        self.eval()
        latent = self.encode(batch)
        return self.head.act(latent, sample=sample)

    # ---- fine-tuning controls [SO-101] ------------------------------------
    def freeze_backbones(self):
        for p in self.region_encoder.trunk.parameters():
            p.requires_grad_(False)
        if self.cfg.wrist_image_key is not None:
            for p in self.context_encoder.wrist_trunk.parameters():
                p.requires_grad_(False)

    def unfreeze_backbones(self):
        for p in self.region_encoder.trunk.parameters():
            p.requires_grad_(True)
        if self.cfg.wrist_image_key is not None:
            for p in self.context_encoder.wrist_trunk.parameters():
                p.requires_grad_(True)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
