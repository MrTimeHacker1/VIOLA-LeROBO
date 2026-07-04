"""Full VIOLA policy (VIOLA Sec. 3).

Pipeline per training step:
  1. For each of the H+1 frames, RegionEncoder produces K region tokens and the
     workspace spatial map; ContextEncoder produces the (global, wrist, proprio)
     context tokens. Together these are the per-step feature h_t.
  2. Temporal composition: add the temporal positional encoding of each frame's
     position to every token of that frame, then concatenate the H+1 per-step
     features into the object-centric representation z_t.
  3. TransformerPolicy reasons over z_t with a prepended action token.
  4. GMMHead maps the action-token latent to a 5-mode Gaussian mixture; loss is
     the NLL of the demonstrated action.

Frame ordering convention (paper-faithful): the window is delivered with
  index 0 = CURRENT frame (h_t), index H = OLDEST frame (h_{t-H}).
Temporal PE index i is added to frame i, so PE_0 tags the current frame and
PE_H tags the oldest -- matching z_t = { h_{t-i} + PE_i }_{i=0..H}. The
demonstrated target is the action at the current frame.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from .config import VIOLAConfig
from .context_encoder import ContextEncoder
from .gmm_head import GMMHead
from .positional_encoding import TemporalPositionalEncoding
from .region_encoder import RegionEncoder
from .transformer_policy import TransformerPolicy

logger = logging.getLogger(__name__)


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

        # temporal PE: index 0 (current) gets PE_0, index H (oldest) gets PE_H.
        tpe = self.temporal_pe.table[:t].to(per_step.dtype)          # [T,D]
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
    def _backbone_param_names(self) -> dict[str, list[str]]:
        """Map backbone name -> list of its parameter names (full-qualified)."""
        out = {"region_encoder.trunk": [
            f"region_encoder.trunk.{n}"
            for n, _ in self.region_encoder.trunk.named_parameters()
        ]}
        if getattr(self.context_encoder, "use_wrist", False):
            out["context_encoder.wrist_trunk"] = [
                f"context_encoder.wrist_trunk.{n}"
                for n, _ in self.context_encoder.wrist_trunk.named_parameters()
            ]
        return out

    def freeze_backbones(self, present_keys: set[str] | None = None):
        """Freeze the workspace (and wrist) ResNet-18 trunks.

        If ``present_keys`` is given (e.g. the keys actually restored from a
        pretrained checkpoint), a trunk is frozen ONLY if it was present -- a
        from-scratch trunk stays trainable so it can still learn.
        """
        named = dict(self.named_parameters())
        for bb, pnames in self._backbone_param_names().items():
            if present_keys is not None and not any(p in present_keys for p in pnames):
                logger.info("Backbone %s not in checkpoint -> left TRAINABLE.", bb)
                continue
            for p in pnames:
                named[p].requires_grad_(False)
            logger.info("Froze backbone %s (%d tensors).", bb, len(pnames))

    def unfreeze_backbones(self):
        for pnames in self._backbone_param_names().values():
            named = dict(self.named_parameters())
            for p in pnames:
                named[p].requires_grad_(True)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---- checkpoint I/O (safetensors, NOT pickled torch.save) --------------
    def save_pretrained(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # safetensors requires contiguous tensors; state_dict tensors are fine.
        state = {k: v.contiguous().cpu() for k, v in self.state_dict().items()}
        save_file(state, str(directory / "model.safetensors"))
        with open(directory / "config.json", "w") as fh:
            json.dump(self.cfg.to_dict(), fh, indent=2)
        return directory

    def load_pretrained_weights(
        self, path: str | Path, strict: bool = False
    ) -> tuple[list[str], list[str]]:
        """Load weights from a .safetensors file (non-strict by default).

        Reports missing / unexpected keys and returns them, plus logs a summary.
        """
        state = load_file(str(path))
        result = self.load_state_dict(state, strict=strict)
        missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
        loaded = [k for k in state.keys() if k not in unexpected]
        logger.info(
            "Loaded %d tensors from %s (missing=%d, unexpected=%d).",
            len(loaded), path, len(missing), len(unexpected),
        )
        if missing:
            logger.info("Missing (fresh-init) keys: %s", missing)
        if unexpected:
            logger.info("Unexpected (ignored) keys: %s", unexpected)
        return missing, unexpected

    @classmethod
    def from_pretrained(
        cls, directory: str | Path, cfg: VIOLAConfig | None = None,
        strict: bool = False,
    ) -> "VIOLAPolicy":
        directory = Path(directory)
        if cfg is None:
            with open(directory / "config.json", "r") as fh:
                cfg = VIOLAConfig.from_dict(json.load(fh))
        model = cls(cfg)
        model.load_pretrained_weights(directory / "model.safetensors", strict=strict)
        return model

    def loaded_keys_from(self, path: str | Path) -> set[str]:
        """Return the set of this model's keys that ``path`` provides (no load)."""
        state = load_file(str(path))
        own = set(self.state_dict().keys())
        return {k for k in state.keys() if k in own}
