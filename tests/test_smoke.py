"""Synthetic smoke test (no network, no detector download).

Builds the policy from config and checks: forward -> finite GMM NLL -> backward;
the transformer sequence length equals 1 + (H+1)*(K+3); select_action returns
[B, action_dim]; freeze/unfreeze toggle requires_grad; ~11M parameters.
"""

from __future__ import annotations

import torch

from viola_so101.config import VIOLAConfig
from viola_so101.policy import VIOLAPolicy


def _synthetic_batch(cfg: VIOLAConfig, b: int = 2) -> dict:
    t, k, s = cfg.n_frames, cfg.num_proposals, cfg.image_size
    # valid normalized xyxy boxes (x1>x0, y1>y0)
    xy0 = torch.rand(b, t, k, 2) * 0.5
    xy1 = xy0 + torch.rand(b, t, k, 2) * 0.5 + 1e-3
    boxes = torch.cat([xy0, xy1], dim=-1).clamp(0, 1)
    batch = {
        "workspace_images": torch.rand(b, t, 3, s, s),
        "wrist_images": torch.rand(b, t, 3, cfg.wrist_image_size, cfg.wrist_image_size),
        "states": torch.rand(b, t, cfg.state_dim),
        "proposals": boxes,
        "action": torch.rand(b, cfg.action_dim),
    }
    return batch


def test_seq_len_formula():
    cfg = VIOLAConfig()
    assert cfg.n_context_tokens == 3          # dual-camera default
    assert cfg.seq_len == 1 + (cfg.history + 1) * (cfg.num_proposals + 3)
    assert cfg.seq_len == 181


def test_param_count_about_11M():
    cfg = VIOLAConfig()
    policy = VIOLAPolicy(cfg)
    total = sum(p.numel() for p in policy.parameters())
    assert 9e6 < total < 13e6, f"expected ~11M params, got {total/1e6:.2f}M"


def test_transformer_sequence_length():
    cfg = VIOLAConfig()
    policy = VIOLAPolicy(cfg).eval()
    captured = {}

    def hook(module, inputs):
        # inputs[0] is the [B, 1+n_obs, D] sequence fed to the encoder
        captured["len"] = inputs[0].shape[1]

    handle = policy.transformer.encoder.register_forward_pre_hook(hook)
    with torch.no_grad():
        policy.encode(_synthetic_batch(cfg, b=1))
    handle.remove()
    assert captured["len"] == cfg.seq_len == 181


def test_forward_backward_finite():
    cfg = VIOLAConfig()
    policy = VIOLAPolicy(cfg).train()
    batch = _synthetic_batch(cfg, b=2)
    loss = policy(batch)
    assert loss.ndim == 0 and torch.isfinite(loss), loss
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads)


def test_select_action_shape():
    cfg = VIOLAConfig()
    policy = VIOLAPolicy(cfg)
    action = policy.select_action(_synthetic_batch(cfg, b=2))
    assert action.shape == (2, cfg.action_dim)


def test_freeze_unfreeze_toggles_requires_grad():
    cfg = VIOLAConfig()
    policy = VIOLAPolicy(cfg)

    def trunk_grads():
        rg = [p.requires_grad for p in policy.region_encoder.trunk.parameters()]
        wg = [p.requires_grad for p in policy.context_encoder.wrist_trunk.parameters()]
        return rg, wg

    rg, wg = trunk_grads()
    assert all(rg) and all(wg)

    policy.freeze_backbones()
    rg, wg = trunk_grads()
    assert not any(rg) and not any(wg)
    # non-backbone params stay trainable
    assert all(p.requires_grad for p in policy.transformer.parameters())

    policy.unfreeze_backbones()
    rg, wg = trunk_grads()
    assert all(rg) and all(wg)


def test_freeze_only_present_backbones():
    cfg = VIOLAConfig()
    policy = VIOLAPolicy(cfg)
    # pretend only the workspace trunk was in the checkpoint
    present = {n for n, _ in policy.named_parameters()
               if n.startswith("region_encoder.trunk")}
    policy.freeze_backbones(present_keys=present)
    assert not any(p.requires_grad for p in policy.region_encoder.trunk.parameters())
    # wrist trunk was absent -> stays trainable
    assert all(p.requires_grad for p in policy.context_encoder.wrist_trunk.parameters())


def test_save_load_roundtrip(tmp_path):
    cfg = VIOLAConfig()
    policy = VIOLAPolicy(cfg)
    policy.save_pretrained(tmp_path / "ckpt")
    reloaded = VIOLAPolicy.from_pretrained(tmp_path / "ckpt")
    missing, unexpected = reloaded.load_pretrained_weights(
        tmp_path / "ckpt" / "model.safetensors", strict=True
    )
    assert not missing and not unexpected


if __name__ == "__main__":
    # allow running directly without pytest
    import tempfile

    test_seq_len_formula()
    test_param_count_about_11M()
    test_transformer_sequence_length()
    test_forward_backward_finite()
    test_select_action_shape()
    test_freeze_unfreeze_toggles_requires_grad()
    test_freeze_only_present_backbones()
    with tempfile.TemporaryDirectory() as d:
        import pathlib
        test_save_load_roundtrip(pathlib.Path(d))
    print("smoke test OK")
