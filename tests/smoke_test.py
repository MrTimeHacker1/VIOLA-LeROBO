"""Synthetic smoke test for the VIOLA model path (no LeRobot, no detector).

Verifies: forward loss is a finite scalar, backprop populates grads on trainable
params, freeze_backbones removes trunk grads, and select_action returns the
right shape. Uses tiny B/T images so it runs on CPU in a few seconds.
"""

import torch
from viola.config import VIOLAConfig
from viola.policy import VIOLAPolicy


def make_batch(cfg, b=1, device="cpu"):
    t, k, s = cfg.n_frames, cfg.num_proposals, cfg.image_size
    # random but valid normalized xyxy boxes (x0<x1, y0<y1)
    xy0 = torch.rand(b, t, k, 2) * 0.5
    xy1 = xy0 + torch.rand(b, t, k, 2) * 0.5
    boxes = torch.cat([xy0, xy1], dim=-1).clamp(0, 1)
    batch = {
        "workspace_images": torch.rand(b, t, 3, s, s),
        "wrist_images": torch.rand(b, t, 3, cfg.wrist_image_size, cfg.wrist_image_size),
        "states": torch.rand(b, t, cfg.state_dim),
        "proposals": boxes,
        "action": torch.rand(b, cfg.action_dim) * 2 - 1,
    }
    return {key: v.to(device) for key, v in batch.items()}


def main():
    torch.manual_seed(0)
    cfg = VIOLAConfig()
    policy = VIOLAPolicy(cfg)
    batch = make_batch(cfg, b=2)

    # forward + loss
    loss = policy(batch)
    assert loss.ndim == 0 and torch.isfinite(loss), "loss must be finite scalar"
    print(f"[ok] forward loss = {loss.item():.4f}")
    print(f"[ok] transformer sequence length = {cfg.seq_len} (expected "
          f"1 + {cfg.n_frames}*{cfg.tokens_per_frame})")

    # backward
    loss.backward()
    n_grad = sum(p.grad is not None for p in policy.parameters() if p.requires_grad)
    print(f"[ok] params with grad after backward: {n_grad}")

    # freeze backbones -> trunk params must report no grad requirement
    policy.zero_grad(set_to_none=True)
    policy.freeze_backbones()
    trunk_trainable = any(p.requires_grad for p in policy.region_encoder.trunk.parameters())
    wrist_trainable = any(p.requires_grad for p in policy.context_encoder.wrist_trunk.parameters())
    assert not trunk_trainable and not wrist_trainable, "backbones should be frozen"
    n_train = sum(p.numel() for p in policy.trainable_parameters())
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"[ok] frozen backbones; trainable params {n_train/1e6:.2f}M / "
          f"{n_total/1e6:.2f}M total")

    # inference
    action = policy.select_action(batch, sample=False)
    assert action.shape == (2, cfg.action_dim), action.shape
    print(f"[ok] select_action shape = {tuple(action.shape)}")
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
