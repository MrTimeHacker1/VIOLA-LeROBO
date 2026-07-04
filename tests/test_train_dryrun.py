"""2-step training dry-run through the real Accelerate loop.

Uses a tiny in-memory synthetic dataset and a FAKE proposal function to exercise
the whole ``train.loop.train`` code path on CPU (single-process): optimizer,
cosine scheduler, gradient clipping, logging, and safetensors checkpointing.
No network, no detector, no GPU required.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from viola_so101.config import VIOLAConfig
from viola_so101.policy import VIOLAPolicy
from viola_so101.train.loop import train


class TinySyntheticDataset(Dataset):
    """Returns samples with the SAME keys/shapes as VIOLALeRobotDataset."""

    def __init__(self, cfg: VIOLAConfig, n: int = 4):
        self.cfg = cfg
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        cfg = self.cfg
        t, s = cfg.n_frames, cfg.image_size
        return {
            "workspace_images": torch.rand(t, 3, s, s),
            "wrist_images": torch.rand(t, 3, cfg.wrist_image_size,
                                       cfg.wrist_image_size),
            "states": torch.rand(t, cfg.state_dim),
            "proposals": torch.zeros(t, cfg.num_proposals, 4),  # filled online
            "action": torch.rand(cfg.action_dim),
            "_has_proposals": torch.tensor(False),
        }


class FakeProposals:
    """Stand-in for ProposalNetwork: returns random valid boxes on CPU."""

    def __init__(self, cfg: VIOLAConfig, device):
        self.cfg = cfg
        self.device = device

    def generate(self, flat: torch.Tensor) -> torch.Tensor:
        n = flat.shape[0]
        xy0 = torch.rand(n, self.cfg.num_proposals, 2) * 0.5
        xy1 = xy0 + torch.rand(n, self.cfg.num_proposals, 2) * 0.5 + 1e-3
        return torch.cat([xy0, xy1], dim=-1).clamp(0, 1)


def test_two_step_accelerate_dry_run(tmp_path):
    # small but real: reduced image / history so it runs fast on CPU, full schema.
    cfg = VIOLAConfig(
        image_size=64, wrist_image_size=64, history=2,
        batch_size=2, epochs=1, num_workers=0, mixed_precision="no",
        log_every=1, output_dir=str(tmp_path),
    )
    # 4 samples, batch 2, drop_last -> exactly 2 optimizer steps.
    dataset = TinySyntheticDataset(cfg, n=4)
    policy = VIOLAPolicy(cfg)

    best_dir = train(
        cfg, policy, dataset,
        run_name="dryrun",
        make_proposal_net=lambda dev: FakeProposals(cfg, dev),
        use_wandb=False,
    )

    # checkpoint written as safetensors + config.json, and reloadable
    assert (best_dir / "model.safetensors").exists()
    assert (best_dir / "config.json").exists()
    reloaded = VIOLAPolicy.from_pretrained(best_dir)
    assert reloaded.cfg.history == cfg.history

    # last checkpoint also written
    assert (best_dir.parent / "last" / "model.safetensors").exists()


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        test_two_step_accelerate_dry_run(Path(d))
    print("train dry-run OK")
