"""Shared training loop for VIOLA pretrain / fine-tune.

Implements the paper's optimisation recipe: AdamW, cosine-annealing LR over the
full epoch budget, gradient clipping, GMM NLL loss, and the paper's checkpoint
criterion (save the checkpoint with the lowest mean training loss).
"""

from __future__ import annotations
import os
import math
import torch
from torch.utils.data import DataLoader

from data.dataset import add_online_proposals


def make_loader(dataset, cfg, shuffle=True, num_workers=4):
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )


def move_to(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def train(policy, loader, cfg, device, out_dir,
          proposal_net=None, online_proposals=False, lr=None):
    os.makedirs(out_dir, exist_ok=True)
    policy.to(device).train()
    lr = cfg.lr if lr is None else lr

    optim = torch.optim.AdamW(
        policy.trainable_parameters(), lr=lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    best = math.inf
    for epoch in range(cfg.epochs):
        running, n = 0.0, 0
        for batch in loader:
            batch = move_to(batch, device)
            if online_proposals:
                batch = add_online_proposals(batch, proposal_net, cfg)

            loss = policy(batch)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), cfg.grad_clip)
            optim.step()

            running += loss.item() * batch["action"].shape[0]
            n += batch["action"].shape[0]
        sched.step()

        mean_loss = running / max(n, 1)
        print(f"epoch {epoch + 1:03d}/{cfg.epochs}  train_nll={mean_loss:.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}")

        torch.save({"model": policy.state_dict(), "cfg": cfg.__dict__,
                    "epoch": epoch}, os.path.join(out_dir, "last.pt"))
        if mean_loss < best:                     # paper's lowest-train-loss rule
            best = mean_loss
            torch.save({"model": policy.state_dict(), "cfg": cfg.__dict__,
                        "epoch": epoch, "train_nll": best},
                       os.path.join(out_dir, "best.pt"))
    print(f"done. best train_nll={best:.4f}  ->  {out_dir}/best.pt")
    return os.path.join(out_dir, "best.pt")
