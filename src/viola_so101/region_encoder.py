"""Region feature encoder (VIOLA Sec. 3.2 + Appendix A/B.2).

A ResNet-18 trunk, trained FROM SCRATCH (random init), encodes the workspace
image into a 16x16 spatial feature map. The paper's Appendix B.2 shows that a
from-scratch spatial map produces more "actionable" features for control than a
frozen/pretrained detection FPN, so we deliberately do not load ImageNet weights.

For each of the K proposal boxes we ROIAlign a 6x6 grid from the spatial map,
flatten, and linearly project to a token. The box positional feature is then
ADDED to that visual feature (paper: "add it to the positional feature of the
same region to obtain a region feature").
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.ops import roi_align

from .config import VIOLAConfig
from .positional_encoding import BoxPositionalEncoding


def build_spatial_trunk() -> nn.Module:
    """ResNet-18 up to and including layer3.

    For a 256x256 input the output is [B, 256, 16, 16] (stride 16), which is the
    16x16 spatial feature map specified in Appendix A. Weights are random
    (``weights=None`` -- the modern torchvision API, never ``pretrained=``)."""
    net = resnet18(weights=None)
    return nn.Sequential(
        net.conv1, net.bn1, net.relu, net.maxpool,
        net.layer1, net.layer2, net.layer3,
    )


class RegionEncoder(nn.Module):
    def __init__(self, cfg: VIOLAConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = build_spatial_trunk()
        roi_dim = cfg.spatial_channels * cfg.roi_output_size * cfg.roi_output_size
        self.visual_proj = nn.Linear(roi_dim, cfg.token_dim)
        self.box_pe = BoxPositionalEncoding(
            cfg.token_dim, image_size=cfg.image_size, base=cfg.pe_base
        )

    def forward(self, image: torch.Tensor, boxes_norm: torch.Tensor):
        """image: [N, 3, S, S]; boxes_norm: [N, K, 4] normalized xyxy.

        Returns:
            region_tokens: [N, K, token_dim]
            fmap:          [N, C, 16, 16]  (reused for the global context token)
        """
        n = image.shape[0]
        k = boxes_norm.shape[1]
        fmap = self.trunk(image)                                   # [N,C,16,16]

        # ROIAlign expects boxes in input-image pixel coords + a spatial_scale.
        boxes_px = boxes_norm * self.cfg.image_size                # [N,K,4]
        batch_idx = torch.arange(n, device=image.device, dtype=image.dtype)
        batch_idx = batch_idx.repeat_interleave(k).unsqueeze(1)    # [N*K,1]
        rois = torch.cat([batch_idx, boxes_px.reshape(n * k, 4)], dim=1)  # [N*K,5]

        pooled = roi_align(
            fmap, rois,
            output_size=(self.cfg.roi_output_size, self.cfg.roi_output_size),
            spatial_scale=1.0 / self.cfg.backbone_stride,
            sampling_ratio=-1,
            aligned=True,
        )                                                          # [N*K,C,6,6]
        visual = self.visual_proj(pooled.flatten(1)).reshape(n, k, -1)
        pos = self.box_pe(boxes_norm)                              # [N,K,token_dim]
        region_tokens = visual + pos
        return region_tokens, fmap
