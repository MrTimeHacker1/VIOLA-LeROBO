"""Offline object-proposal cache builder (entry point: ``viola-preprocess``).

Runs the frozen Faster R-CNN RPN once over every frame of a LeRobotDataset in
global order and writes a memmap ``[N, K, 4]`` of normalized xyxy boxes aligned
1:1 with frame order. Training then reads the cache and never runs the detector.
Single-process job; the output path must be user-writable.

    viola-preprocess --repo-id lerobot/svla_so100_stacking --out /data/stacking.npy
"""

from __future__ import annotations

import argparse
import logging

import torch
from torchvision.transforms import v2

from .config import VIOLAConfig
from .logging_utils import setup_logging
from .proposals import ProposalNetwork

logger = logging.getLogger(__name__)


def _frame_iterator(ds, cfg: VIOLAConfig):
    """Yield each frame's workspace image as float CHW resized to image_size,
    in global order 0..N-1 (clean frames -- no augmentation)."""
    for i in range(len(ds)):
        img = ds[i][cfg.workspace_image_key]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        img = img.float()
        yield v2.functional.resize(img, [cfg.image_size, cfg.image_size],
                                   antialias=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an offline proposal cache")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--out", required=True, help="output .npy memmap path")
    ap.add_argument("--config", default=None, help="YAML config (optional)")
    ap.add_argument("--root", default=None, help="local dataset root (optional)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    setup_logging(level=None)
    cfg = VIOLAConfig.from_yaml(args.config) if args.config else VIOLAConfig()

    from lerobot.datasets import LeRobotDataset  # [version seam]
    ds = LeRobotDataset(args.repo_id, root=args.root)
    logger.info("Preprocessing %s: %d frames -> %s (device=%s, K=%d)",
                args.repo_id, len(ds), args.out, args.device, cfg.num_proposals)

    net = ProposalNetwork(cfg.num_proposals, device=args.device)
    net.precompute_dataset(
        _frame_iterator(ds, cfg), num_frames=len(ds),
        cache_path=args.out, batch_size=args.batch_size,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
