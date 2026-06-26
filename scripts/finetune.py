"""Stage 2: fine-tune the pretrained VIOLA policy on your SO-101 episodes.

Loads the Stage-1 checkpoint, (optionally) freezes both ResNet-18 backbones so
the learned visual representation is preserved, and continues training at a
lower learning rate on your recorded stacking demonstrations. Proposals are run
online by default since the fine-tune set is small.

Example:
    python scripts/finetune.py \
        --repo-id <your-hf-user>/so101_stacking \
        --pretrained runs/pretrain/best.pt \
        --out runs/finetune --device cuda
"""

from __future__ import annotations
import argparse
import torch

from viola.config import VIOLAConfig
from viola.policy import VIOLAPolicy
from viola.proposals import ProposalNetwork
from data.dataset import VIOLALeRobotDataset
from scripts.train_common import train, make_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="your SO-101 stacking dataset")
    ap.add_argument("--pretrained", required=True, help="Stage-1 checkpoint .pt")
    ap.add_argument("--cache", default=None, help="optional offline proposal .npy")
    ap.add_argument("--out", default="runs/finetune")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-freeze", action="store_true",
                    help="fine-tune backbones too (default: frozen)")
    ap.add_argument("--no-wrist", action="store_true")
    args = ap.parse_args()

    cfg = VIOLAConfig.for_finetune()
    if args.no_wrist:
        cfg.wrist_image_key = None

    policy = VIOLAPolicy(cfg)
    ckpt = torch.load(args.pretrained, map_location="cpu")
    policy.load_state_dict(ckpt["model"])
    if cfg.freeze_backbones_on_finetune and not args.no_freeze:
        policy.freeze_backbones()
        print("froze workspace + wrist ResNet-18 backbones")

    online = args.cache is None
    dataset = VIOLALeRobotDataset(args.repo_id, cfg, train=True,
                                  proposal_cache_path=args.cache)
    loader = make_loader(dataset, cfg, shuffle=True, num_workers=args.num_workers)

    proposal_net = ProposalNetwork(cfg.num_proposals, device=args.device) if online else None

    train(policy, loader, cfg, args.device, args.out,
          proposal_net=proposal_net, online_proposals=online, lr=cfg.finetune_lr)


if __name__ == "__main__":
    main()
