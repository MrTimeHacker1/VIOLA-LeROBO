"""Stage 1: pretrain VIOLA from scratch on pooled SO-100/101 community data.

Trains the full policy (random init) on one or more LeRobotDatasets that share
the SO-101 6-DoF joint action/state schema, using offline proposal caches.

Example:
    python scripts/pretrain.py \
        --repo-id lerobot/svla_so100_stacking lerobot/svla_so101_pickplace \
        --cache caches/svla_so100_stacking.npy caches/svla_so101_pickplace.npy \
        --out runs/pretrain --device cuda
"""

from __future__ import annotations
import argparse
import torch
from torch.utils.data import ConcatDataset

from viola.config import VIOLAConfig
from viola.policy import VIOLAPolicy
from viola.proposals import ProposalNetwork
from data.dataset import VIOLALeRobotDataset
from scripts.train_common import train, make_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", nargs="+", required=True)
    ap.add_argument("--cache", nargs="*", default=[],
                    help="offline proposal .npy per repo-id (recommended)")
    ap.add_argument("--out", default="runs/pretrain")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-wrist", action="store_true",
                    help="disable the eye-in-hand token (single-camera setups)")
    args = ap.parse_args()

    cfg = VIOLAConfig.for_pretrain()
    if args.no_wrist:
        cfg.wrist_image_key = None

    caches = args.cache + [None] * (len(args.repo_id) - len(args.cache))
    online = any(c is None for c in caches)

    datasets = [VIOLALeRobotDataset(rid, cfg, train=True, proposal_cache_path=c)
                for rid, c in zip(args.repo_id, caches)]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    loader = make_loader(dataset, cfg, shuffle=True, num_workers=args.num_workers)

    policy = VIOLAPolicy(cfg)
    proposal_net = ProposalNetwork(cfg.num_proposals, device=args.device) if online else None
    if online:
        print("WARNING: no cache for some datasets -> running proposals online "
              "(set --num-workers 0 recommended).")

    train(policy, loader, cfg, args.device, args.out,
          proposal_net=proposal_net, online_proposals=online, lr=cfg.lr)


if __name__ == "__main__":
    main()
