"""Positional encodings (VIOLA Appendix A).

Box positional encoding (region "where"):
    For a box with corners (x0, y0, x1, y1), dimension j in [0, D):
        coord    = [x0, y0, x1, y1][j % 4]      # in PIXEL units
        angle    = coord / (base ** (j / D))
        PE[j]    = sin(angle)  if (j // 4) is even
                   cos(angle)  if (j // 4) is odd
    The paper feeds pixel coordinates (not normalized) so the sinusoids have a
    usable dynamic range; we therefore multiply normalized boxes by image_size.
    base = 10 (paper). The box PE is ADDED to the region's visual feature.

Temporal positional encoding (frame order):
    Standard sinusoidal table over positions {0..H}, base = 10 (paper). Applied
    with index 0 = CURRENT frame (h_t, PE_0) and index H = OLDEST frame
    (h_{t-H}, PE_H), matching the paper's z_t = { h_{t-i} + PE_i }_{i=0..H}.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BoxPositionalEncoding(nn.Module):
    def __init__(self, dim: int, image_size: int, base: float = 10.0):
        super().__init__()
        assert dim % 4 == 0
        self.dim = dim
        self.image_size = float(image_size)
        # divisor[j] = base ** (j / D), one per output dimension
        j = torch.arange(dim, dtype=torch.float32)
        divisor = base ** (j / dim)
        self.register_buffer("divisor", divisor, persistent=False)  # [D]
        # which corner each output dim reads, and whether sin or cos is used
        coord_idx = (j % 4).long()                       # 0..3 -> x0,y0,x1,y1
        use_sin = (torch.div(j, 4, rounding_mode="floor") % 2 == 0)
        self.register_buffer("coord_idx", coord_idx, persistent=False)  # [D]
        self.register_buffer("use_sin", use_sin, persistent=False)      # [D] bool

    def forward(self, boxes_norm: torch.Tensor) -> torch.Tensor:
        # boxes_norm: [..., 4] normalized xyxy in [0,1]
        coords_px = boxes_norm * self.image_size                 # [..., 4]
        # pick the corner coordinate for each output dimension -> [..., D]
        selected = coords_px[..., self.coord_idx]
        angles = selected / self.divisor                          # [..., D]
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return torch.where(self.use_sin, sin, cos)                # [..., D]


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, n_positions: int, base: float = 10.0):
        super().__init__()
        pe = torch.zeros(n_positions, dim)
        pos = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)  # [P,1]
        i = torch.arange(dim, dtype=torch.float32).unsqueeze(0)            # [1,D]
        divisor = base ** (i / dim)
        angles = pos / divisor                                            # [P,D]
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        self.register_buffer("pe", pe, persistent=False)                  # [P,D]

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: long tensor of frame indices -> [..., D]
        return self.pe[positions]

    @property
    def table(self) -> torch.Tensor:
        return self.pe
