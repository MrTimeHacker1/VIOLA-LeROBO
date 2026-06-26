"""General object proposals (VIOLA Sec. 3.2).

The paper uses a frozen, pretrained Region Proposal Network (Detic / Faster
R-CNN family) to produce class-agnostic "objectness" boxes from natural-image
supervision. We use torchvision's pretrained Faster R-CNN and read proposals
straight from its RPN, which is the direct architectural ancestor of the
paper's detector and requires no detectron2 dependency.

This network is ALWAYS frozen and ALWAYS run under torch.no_grad(). It is never
part of the policy's trainable parameters at any stage.

Output convention: top-K boxes per image as normalized xyxy in [0, 1].
"""

from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)


class ProposalNetwork(nn.Module):
    def __init__(self, num_proposals: int = 15, device: str = "cpu"):
        super().__init__()
        self.num_proposals = num_proposals
        self.device = device
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        model.eval()
        # Ask the RPN for more candidates than we keep, so top-K is meaningful.
        model.rpn.post_nms_top_n = lambda: max(256, num_proposals * 8)
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.to(device)

    @torch.no_grad()
    def generate(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, H, W] float in [0, 1].

        Returns boxes_norm: [B, K, 4] normalized xyxy in [0, 1], ordered by
        descending objectness. If the RPN returns fewer than K boxes, the set is
        padded with the full-image box [0, 0, 1, 1]."""
        images = images.to(self.device)
        image_list = [img for img in images]                      # list of [3,H,W]
        transformed, _ = self.model.transform(image_list)
        features = self.model.backbone(transformed.tensors)
        proposals, _ = self.model.rpn(transformed, features)      # list of [N_i,4]

        out = []
        for boxes, (h, w) in zip(proposals, transformed.image_sizes):
            boxes = boxes[: self.num_proposals]
            scale = torch.tensor([w, h, w, h], device=boxes.device, dtype=boxes.dtype)
            boxes = (boxes / scale).clamp(0.0, 1.0)               # normalize
            if boxes.shape[0] < self.num_proposals:
                pad = boxes.new_tensor([[0.0, 0.0, 1.0, 1.0]])
                pad = pad.repeat(self.num_proposals - boxes.shape[0], 1)
                boxes = torch.cat([boxes, pad], dim=0)
            out.append(boxes)
        return torch.stack(out, dim=0).cpu()                      # [B,K,4]

    @torch.no_grad()
    def precompute_dataset(self, frame_iterator, num_frames: int,
                           cache_path: str, image_size: int = 256,
                           batch_size: int = 32) -> str:
        """Run proposals over a dataset once and store a memmap [N, K, 4].

        `frame_iterator` must yield workspace images as float CHW tensors in
        [0,1], already resized to (image_size, image_size), in global frame
        order 0..num_frames-1. The resulting .npy aligns 1:1 with that order.
        """
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        cache = np.lib.format.open_memmap(
            cache_path, mode="w+", dtype=np.float32,
            shape=(num_frames, self.num_proposals, 4),
        )
        buf, idx0 = [], 0
        for i, img in enumerate(frame_iterator):
            buf.append(img)
            if len(buf) == batch_size or i == num_frames - 1:
                batch = torch.stack(buf, dim=0)
                boxes = self.generate(batch).numpy()
                cache[idx0: idx0 + boxes.shape[0]] = boxes
                idx0 += boxes.shape[0]
                buf = []
        cache.flush()
        return cache_path
