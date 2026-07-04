"""Data augmentation (VIOLA Appendix A).

Applied to float CHW image tensors in [0, 1].

  color jitter   : brightness/contrast/saturation 0.3, hue 0.05, on 90% of
                   samples. The same jitter params are SHARED across the H+1
                   frames of one window to preserve temporal coherence; 10% of
                   windows are left unchanged.
  pixel shift    : random translation up to 4 px, SHARED across the window.
  random erasing : p=0.5, scale (0.02,0.05), ratio (0.5,1.5), gaussian fill,
                   applied INDEPENDENTLY per workspace frame.
  box noise      : small gaussian noise on normalized proposal coordinates.

Symmetry fix vs. the prior implementation: the pixel shift and the proposal
boxes are kept CONSISTENT. When boxes come from an offline cache (computed on
clean frames), the same shift applied to the image is also applied to the boxes,
so region features do not desync from the shifted pixels. In the online path the
detector runs on the already-shifted image, so boxes are consistent by
construction; only box noise is added there (see ``add_online_proposals``).

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
            value="random",   # gaussian-ish random fill
        )

    # -- sample shared params for one window --------------------------------
    def sample_params(self) -> dict:
        """Sample the color-jitter + pixel-shift params shared across a window."""
        color = None
        if torch.rand(()) <= self.cfg.color_jitter_prob:
            # (fn_idx, brightness, contrast, saturation, hue)
            color = v2.ColorJitter.get_params(
                self.color.brightness, self.color.contrast,
                self.color.saturation, self.color.hue,
            )
        dx = dy = 0
        m = self.cfg.pixel_shift
        if m > 0:
            dx = int(torch.randint(-m, m + 1, ()).item())
            dy = int(torch.randint(-m, m + 1, ()).item())
        return {"color": color, "shift": (dx, dy)}

    # -- apply to an image window -------------------------------------------
    def _apply_color(self, frames: torch.Tensor, color) -> torch.Tensor:
        if color is None:
            return frames
        fn_idx, b, c, s, h = color
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

    def _apply_shift(self, frames: torch.Tensor, shift) -> torch.Tensor:
        dx, dy = shift
        if dx == 0 and dy == 0:
            return frames
        return v2.functional.affine(
            frames, angle=0.0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0]
        )

    def apply_image(self, frames: torch.Tensor, params: dict,
                    erase: bool) -> torch.Tensor:
        """frames: [T,3,H,W]. ``erase`` toggles per-frame random erasing
        (True for workspace frames, False for wrist frames)."""
        frames = self._apply_color(frames, params["color"])
        frames = self._apply_shift(frames, params["shift"])
        if erase:
            frames = torch.stack([self.eraser(f) for f in frames], dim=0)
        return frames

    # -- apply to boxes consistently with the same window params ------------
    def apply_boxes(self, boxes: torch.Tensor, params: dict,
                    shift: bool = True) -> torch.Tensor:
        """boxes: [T,K,4] normalized xyxy. Shift boxes by the same pixel delta
        as the image (when ``shift``), then add small gaussian noise."""
        if shift:
            dx, dy = params["shift"]
            if dx or dy:
                delta = boxes.new_tensor([dx, dy, dx, dy]) / float(self.cfg.image_size)
                boxes = (boxes + delta).clamp(0.0, 1.0)
        return self.noise_boxes(boxes)

    def noise_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        if self.cfg.box_noise_std <= 0:
            return boxes
        noise = torch.randn_like(boxes) * self.cfg.box_noise_std
        return (boxes + noise).clamp(0.0, 1.0)
