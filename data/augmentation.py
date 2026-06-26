"""Data augmentation (VIOLA Appendix A).

Applied to float CHW image tensors in [0, 1].

  color jitter   : brightness/contrast/saturation 0.3, hue 0.05, on 90% of
                   samples. The same jitter params are shared across the H+1
                   frames of one sample to preserve temporal coherence; 10% of
                   samples are left unchanged.
  pixel shift    : random translation up to 4 px, shared across the window.
  random erasing : p=0.5, scale (0.02,0.05), ratio (0.5,1.5), gaussian fill,
                   applied INDEPENDENTLY per workspace frame (the paper applies
                   it per training step to prevent overfitting to proposals).
  box noise      : small gaussian noise on normalized proposal coordinates.

Uses torchvision.transforms.v2 (current API).
"""

from __future__ import annotations
import torch
from torchvision.transforms import v2


class VIOLAAugmentation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.color = v2.ColorJitter(
            brightness=cfg.color_jitter_brightness,
            contrast=cfg.color_jitter_contrast,
            saturation=cfg.color_jitter_saturation,
            hue=cfg.color_jitter_hue,
        )
        self.eraser = v2.RandomErasing(
            p=cfg.random_erase_prob,
            scale=cfg.random_erase_scale,
            ratio=cfg.random_erase_ratio,
            value="random",
        )

    # -- shared color jitter across a window -------------------------------
    def _apply_shared_color(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [T,3,H,W]; one sampled transform reused for all T frames
        if torch.rand(()) > self.cfg.color_jitter_prob:
            return frames
        # Sample jitter factors ONCE and apply the same functional ops to every
        # frame in the window, so the temporal sequence stays photometrically
        # coherent.
        fn_idx, b, c, s, h = v2.ColorJitter.get_params(
            self.color.brightness, self.color.contrast,
            self.color.saturation, self.color.hue,
        )
        out = frames
        for fn_id in fn_idx:
            if fn_id == 0 and b is not None:
                out = v2.functional.adjust_brightness(out, b)
            elif fn_id == 1 and c is not None:
                out = v2.functional.adjust_contrast(out, c)
            elif fn_id == 2 and s is not None:
                out = v2.functional.adjust_saturation(out, s)
            elif fn_id == 3 and h is not None:
                out = v2.functional.adjust_hue(out, h)
        return out

    # -- shared pixel shift across a window --------------------------------
    def _apply_shared_shift(self, frames: torch.Tensor) -> torch.Tensor:
        m = self.cfg.pixel_shift
        if m <= 0:
            return frames
        dx = int(torch.randint(-m, m + 1, ()).item())
        dy = int(torch.randint(-m, m + 1, ()).item())
        if dx == 0 and dy == 0:
            return frames
        return v2.functional.affine(
            frames, angle=0.0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0]
        )

    def augment_window(self, frames: torch.Tensor, erase: bool) -> torch.Tensor:
        """frames: [T,3,H,W]. `erase` controls per-frame random erasing
        (True for workspace frames, False for wrist frames)."""
        frames = self._apply_shared_color(frames)
        frames = self._apply_shared_shift(frames)
        if erase:
            frames = torch.stack([self.eraser(f) for f in frames], dim=0)
        return frames

    def augment_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """boxes: [T,K,4] normalized. Add small gaussian noise, keep valid."""
        if self.cfg.box_noise_std <= 0:
            return boxes
        noise = torch.randn_like(boxes) * self.cfg.box_noise_std
        return (boxes + noise).clamp(0.0, 1.0)
