"""Stage-1 helper: precompute object proposals for a LeRobotDataset, once.

Runs the frozen proposal network over every frame of a dataset (in global frame
order) and writes a memmap [N, K, 4] of normalized xyxy boxes that aligns 1:1
with the dataset's frame indexing. Training then loads boxes from this cache
instead of running the detector every step.

Usage:
    python scripts/preprocess.py --repo-id lerobot/svla_so100_stacking \
        --out caches/svla_so100_stacking.npy --device cuda
"""

from __future__ import annotations
import argparse
import torch
from torchvision.transforms import v2

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # [version seam]
from viola.config import VIOLAConfig
from viola.proposals import ProposalNetwork
from data.dataset import _to_float_chw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    cfg = VIOLAConfig()
    ds = LeRobotDataset(args.repo_id)
    n = len(ds)
    net = ProposalNetwork(num_proposals=cfg.num_proposals, device=args.device)

    def frames():
        for g in range(n):
            img = _to_float_chw(ds[g][cfg.workspace_image_key])
            yield v2.functional.resize(img, [cfg.image_size, cfg.image_size], antialias=True)

    print(f"precomputing proposals for {n} frames of {args.repo_id} ...")
    net.precompute_dataset(frames(), num_frames=n, cache_path=args.out,
                           image_size=cfg.image_size, batch_size=args.batch_size)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
