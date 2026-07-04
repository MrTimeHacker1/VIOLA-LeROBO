"""Shared training loop for VIOLA pretrain / fine-tune (Hugging Face Accelerate).

Implements the paper's optimisation recipe: AdamW, cosine-annealing LR over the
full epoch budget, gradient clipping (0.1), GMM NLL loss, and the paper's
checkpoint criterion (save the checkpoint with the lowest mean TRAINING loss).

A single loop serves both stages; the only differences the caller supplies are
data, learning rate, which weights are frozen, and whether proposals come from
an offline cache (dataset) or an online detector (``make_proposal_net``).

All device placement, DDP and mixed precision go through ``Accelerator`` so the
same code path runs on CPU, one GPU, or many GPUs launched with
``accelerate launch``. Logging / checkpointing / wandb happen on the main
process only.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from ..config import VIOLAConfig
from ..data.dataset import add_online_proposals
from ..logging_utils import setup_logging

logger = logging.getLogger(__name__)


def train(
    cfg: VIOLAConfig,
    policy,
    dataset,
    *,
    run_name: str = "viola",
    make_proposal_net=None,          # callable(device) -> net with .generate(); None => offline cache
    use_wandb: bool = False,
    wandb_project: str = "viola-so101",
    push_to_hub: str | None = None,
    aug=None,                        # augmentation instance for online box noise
) -> Path:
    out_dir = Path(cfg.output_dir) / run_name
    online_proposals = make_proposal_net is not None

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with="wandb" if use_wandb else None,
    )
    set_seed(cfg.seed)
    setup_logging(out_dir, is_main_process=accelerator.is_main_process)

    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=== VIOLA-SO101 training: %s ===", run_name)
        logger.info("processes=%d device=%s mixed_precision=%s",
                    accelerator.num_processes, accelerator.device,
                    cfg.mixed_precision)
        logger.info("seq_len=%d tokens/frame=%d n_context=%d wrist=%s",
                    cfg.seq_len, cfg.tokens_per_frame, cfg.n_context_tokens,
                    cfg.wrist_image_key)
        n_train = sum(p.numel() for p in policy.trainable_parameters())
        n_total = sum(p.numel() for p in policy.parameters())
        logger.info("params: %.2fM total, %.2fM trainable",
                    n_total / 1e6, n_train / 1e6)
        logger.info("config: %s", cfg.to_dict())

    if use_wandb:
        accelerator.init_trackers(wandb_project, config=cfg.to_dict(),
                                  init_kwargs={"wandb": {"name": run_name}})

    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    optimizer = torch.optim.AdamW(
        policy.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    policy, optimizer, loader, scheduler = accelerator.prepare(
        policy, optimizer, loader, scheduler
    )

    proposal_net = None
    if online_proposals:
        proposal_net = make_proposal_net(accelerator.device)

    best = math.inf
    best_epoch = -1
    global_step = 0
    try:
        for epoch in range(cfg.epochs):
            policy.train()
            run_sum = torch.zeros((), device=accelerator.device)
            run_cnt = torch.zeros((), device=accelerator.device)

            for batch in loader:
                with accelerator.accumulate(policy):
                    if online_proposals:
                        batch = add_online_proposals(
                            batch, proposal_net, cfg,
                            train=True, aug=aug,
                        )
                    loss = policy(batch)
                    accelerator.backward(loss)
                    grad_norm = None
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(
                            policy.parameters(), cfg.grad_clip
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                bs = batch["action"].shape[0]
                run_sum += loss.detach() * bs
                run_cnt += bs

                if accelerator.sync_gradients:
                    global_step += 1
                    if global_step % cfg.log_every == 0:
                        log_d = {
                            "train/loss": loss.detach().item(),
                            "train/lr": scheduler.get_last_lr()[0],
                        }
                        if grad_norm is not None:
                            log_d["train/grad_norm"] = float(grad_norm)
                        accelerator.log(log_d, step=global_step)

            scheduler.step()

            # global mean training loss across all processes
            tot = accelerator.reduce(run_sum, reduction="sum")
            cnt = accelerator.reduce(run_cnt, reduction="sum")
            mean_loss = (tot / cnt.clamp(min=1)).item()

            if accelerator.is_main_process:
                logger.info("epoch %03d/%d  train_nll=%.4f  lr=%.2e",
                            epoch + 1, cfg.epochs, mean_loss,
                            scheduler.get_last_lr()[0])
                accelerator.log({"train/epoch_nll": mean_loss}, step=global_step)

                unwrapped = accelerator.unwrap_model(policy)
                unwrapped.save_pretrained(out_dir / "last")
                if mean_loss < best:            # paper's lowest-train-loss rule
                    best, best_epoch = mean_loss, epoch
                    unwrapped.save_pretrained(out_dir / "best")

            accelerator.wait_for_everyone()
    finally:
        if accelerator.is_main_process and use_wandb:
            accelerator.log({"best_train_nll": best, "best_epoch": best_epoch},
                            step=global_step)
        accelerator.end_training()

    if accelerator.is_main_process:
        logger.info("done. best train_nll=%.4f (epoch %d) -> %s",
                    best, best_epoch + 1, out_dir / "best")
        if push_to_hub:
            _push_to_hub(out_dir / "best", push_to_hub)
    return out_dir / "best"


def _push_to_hub(local_dir: Path, repo_id: str) -> None:
    """Upload the best checkpoint folder using the current huggingface_hub API
    (HfApi.create_repo + upload_folder). Never uses the deprecated Repository."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id,
                      repo_type="model")
    logger.info("Pushed %s -> https://huggingface.co/%s", local_dir, repo_id)
