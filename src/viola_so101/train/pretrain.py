"""Stage-1 pretraining CLI (entry point: ``viola-pretrain``).

Trains the full policy from random init on the pooled SO-100 datasets, WITH the
wrist token (symmetric dual-camera). Launch with ``accelerate launch``:

    accelerate launch -m viola_so101.train.pretrain \
        --repo-id lerobot/svla_so100_stacking lerobot/svla_so100_sorting \
        --cache /data/stacking.npy /data/sorting.npy

If ``--cache`` is omitted, proposals are generated online on-device.
"""

from __future__ import annotations

import argparse
import logging

from ..config import VIOLAConfig
from ..data.dataset import build_pooled_dataset
from ..logging_utils import setup_logging
from ..policy import VIOLAPolicy
from ..proposals import ProposalNetwork
from .loop import train

logger = logging.getLogger(__name__)


def _add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", type=str, default=None, help="YAML config to load")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--frame-stride", type=int, default=None)
    ap.add_argument("--mixed-precision", choices=["bf16", "fp16", "no"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-wrist", action="store_true",
                    help="disable the wrist token (single-camera; NOT recommended)")
    ap.add_argument("--wandb", dest="wandb", action="store_true")
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.set_defaults(wandb=False)
    ap.add_argument("--wandb-project", type=str, default="viola-so101")
    ap.add_argument("--push-to-hub", type=str, default=None,
                    help="repo_id to upload the best checkpoint to")


def _build_cfg(args, factory) -> VIOLAConfig:
    base = VIOLAConfig.from_yaml(args.config) if args.config else VIOLAConfig()
    cfg = base.merged(
        output_dir=args.output_dir, epochs=args.epochs,
        batch_size=args.batch_size, num_workers=args.num_workers,
        frame_stride=args.frame_stride, mixed_precision=args.mixed_precision,
        seed=args.seed,
    )
    if args.no_wrist:
        cfg.wrist_image_key = None
    # re-apply the stage's lr rule, honouring an explicit --lr override
    cfg = factory(**cfg.to_dict())
    if args.lr is not None:
        cfg.lr = args.lr
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="VIOLA-SO101 Stage-1 pretraining")
    ap.add_argument("--repo-id", nargs="+", required=True,
                    help="one or more LeRobotDataset repo_ids to pool")
    ap.add_argument("--cache", nargs="*", default=None,
                    help="offline proposal cache .npy per repo (aligned order)")
    _add_common_args(ap)
    args = ap.parse_args()

    setup_logging(level=None)
    cfg = _build_cfg(args, VIOLAConfig.for_pretrain)

    caches = args.cache if args.cache else None
    if caches is not None and len(caches) != len(args.repo_id):
        raise SystemExit("--cache must provide one path per --repo-id (or be omitted)")

    dataset = build_pooled_dataset(args.repo_id, cfg, caches=caches, train=True)
    policy = VIOLAPolicy(cfg)

    make_proposal_net = None
    if caches is None:
        logger.info("No cache given -> online proposals on-device.")
        make_proposal_net = lambda dev: ProposalNetwork(cfg.num_proposals, device=dev)

    train(cfg, policy, dataset, run_name="pretrain",
          make_proposal_net=make_proposal_net,
          use_wandb=args.wandb, wandb_project=args.wandb_project,
          push_to_hub=args.push_to_hub)


if __name__ == "__main__":
    main()
