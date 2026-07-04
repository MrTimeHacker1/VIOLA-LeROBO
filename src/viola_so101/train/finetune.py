"""Stage-2 fine-tuning CLI (entry point: ``viola-finetune``).

Loads a Stage-1 checkpoint (non-strict, reporting missing/unexpected keys),
freezes BOTH ResNet-18 backbones by default (only those actually present in the
checkpoint), and trains the rest at lr 1e-5 on the user's SO-101 demos, WITH the
wrist token. Proposals are generated online by default.

    accelerate launch -m viola_so101.train.finetune \
        --repo-id <your/so101_demos> --pretrained ~/.cache/viola_so101/runs/pretrain/best
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..config import VIOLAConfig
from ..data.dataset import build_pooled_dataset
from ..logging_utils import setup_logging
from ..policy import VIOLAPolicy
from ..proposals import ProposalNetwork
from .loop import train
from .pretrain import _add_common_args, _build_cfg

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="VIOLA-SO101 Stage-2 fine-tuning")
    ap.add_argument("--repo-id", nargs="+", required=True,
                    help="LeRobotDataset repo_id(s) with your SO-101 demos")
    ap.add_argument("--pretrained", required=True,
                    help="Stage-1 checkpoint dir (contains model.safetensors)")
    ap.add_argument("--cache", nargs="*", default=None,
                    help="optional offline proposal cache per repo")
    ap.add_argument("--no-freeze", action="store_true",
                    help="do NOT freeze the backbones (override the default)")
    _add_common_args(ap)
    args = ap.parse_args()

    setup_logging(level=None)
    cfg = _build_cfg(args, VIOLAConfig.for_finetune)
    if args.no_freeze:
        cfg.freeze_backbones_on_finetune = False

    dataset = build_pooled_dataset(args.repo_id, cfg,
                                   caches=args.cache or None, train=True)

    policy = VIOLAPolicy(cfg)
    weights = Path(args.pretrained) / "model.safetensors"
    if not weights.exists():                    # allow passing the file directly
        weights = Path(args.pretrained)
    present = policy.loaded_keys_from(weights)
    policy.load_pretrained_weights(weights, strict=False)

    if cfg.freeze_backbones_on_finetune:
        # freeze only backbones that were actually restored (a from-scratch trunk
        # stays trainable). Default symmetric path => both present & frozen.
        policy.freeze_backbones(present_keys=present)

    make_proposal_net = None
    if not args.cache:
        logger.info("No cache given -> online proposals on-device.")
        make_proposal_net = lambda dev: ProposalNetwork(cfg.num_proposals, device=dev)

    train(cfg, policy, dataset, run_name="finetune",
          make_proposal_net=make_proposal_net,
          use_wandb=args.wandb, wandb_project=args.wandb_project,
          push_to_hub=args.push_to_hub)


if __name__ == "__main__":
    main()
