"""LeRobot dataset wrapper for VIOLA (temporal windows + proposals).

This wraps a `LeRobotDataset` and, for each center frame, builds the H+1 frame
window the policy consumes. Windows are constructed by explicit global indexing
(not delta_timestamps) so that offline proposal caches align 1:1 with frames.

Frame ordering in the returned window: index 0 = oldest (t-H*stride),
index H = current (t). Windows are clamped/padded at the start of an episode by
repeating the episode's first frame.

NOTE [version seam]: the LeRobot import path and per-frame metadata columns
('episode_index', etc.) follow the 2026 LeRobot v3 API. If your installed
version differs, adjust the import and the `_episode_starts` helper only.
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .augmentation import VIOLAAugmentation


def _load_lerobot_dataset(repo_id: str):
    # Lazy import so the model/training utilities don't require lerobot. [seam]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id)


def _to_float_chw(img: torch.Tensor) -> torch.Tensor:
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    return img


def _episode_starts(ds) -> np.ndarray:
    """Return, for every global frame, the global start index of its episode."""
    ep = np.asarray(ds.hf_dataset["episode_index"])
    counts = np.bincount(ep)
    starts = np.cumsum(counts) - counts          # start index per episode id
    return starts[ep]                            # [N]


class VIOLALeRobotDataset(Dataset):
    def __init__(self, repo_id_or_dataset, cfg, train: bool = True,
                 proposal_cache_path: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.train = train
        if isinstance(repo_id_or_dataset, str):
            self.ds = _load_lerobot_dataset(repo_id_or_dataset)
        else:
            self.ds = repo_id_or_dataset
        self.n = len(self.ds)
        self.ep_start = _episode_starts(self.ds)
        self.aug = VIOLAAugmentation(cfg) if train else None
        self.proposals = (
            np.load(proposal_cache_path, mmap_mode="r")
            if proposal_cache_path is not None else None
        )
        if self.proposals is not None:
            assert self.proposals.shape[0] == self.n, "proposal cache misaligned"

    def __len__(self):
        return self.n

    def _window_indices(self, center: int) -> list[int]:
        s = int(self.ep_start[center])
        st = self.cfg.frame_stride
        H = self.cfg.history
        return [max(s, center - (H - p) * st) for p in range(H + 1)]

    def _resize(self, img: torch.Tensor, size: int) -> torch.Tensor:
        return v2.functional.resize(img, [size, size], antialias=True)

    def __getitem__(self, center: int):
        cfg = self.cfg
        idxs = self._window_indices(center)

        ws, wr, states = [], [], []
        for g in idxs:
            frame = self.ds[g]
            w_img = _to_float_chw(frame[cfg.workspace_image_key])
            ws.append(self._resize(w_img, cfg.image_size))
            if cfg.wrist_image_key is not None:
                e_img = _to_float_chw(frame[cfg.wrist_image_key])
                wr.append(self._resize(e_img, cfg.wrist_image_size))
            states.append(frame[cfg.state_key].float())

        ws = torch.stack(ws, dim=0)                                # [T,3,S,S]
        states = torch.stack(states, dim=0)                       # [T,state_dim]
        action = self.ds[center][cfg.action_key].float()          # [action_dim]
        wr = torch.stack(wr, dim=0) if cfg.wrist_image_key else None

        if self.proposals is not None:
            boxes = torch.from_numpy(np.asarray(self.proposals[idxs])).float()
        else:
            # filled later by add_online_proposals(); placeholder keeps shape
            boxes = torch.zeros(cfg.n_frames, cfg.num_proposals, 4)

        if self.train and self.aug is not None:
            ws = self.aug.augment_window(ws, erase=True)
            if wr is not None:
                wr = self.aug.augment_window(wr, erase=False)
            if self.proposals is not None:
                boxes = self.aug.augment_boxes(boxes)

        sample = {
            "workspace_images": ws,
            "states": states,
            "proposals": boxes,
            "action": action,
            "_has_proposals": torch.tensor(self.proposals is not None),
        }
        if wr is not None:
            sample["wrist_images"] = wr
        return sample


@torch.no_grad()
def add_online_proposals(batch: dict, proposal_net, cfg) -> dict:
    """Fill `batch['proposals']` by running the frozen proposal net on the GPU.

    Used when no offline cache exists (e.g. fine-tuning). Operates on the
    workspace frames of the whole batch at once. Should be called inside the
    training loop AFTER moving the batch to the device."""
    ws = batch["workspace_images"]                                # [B,T,3,S,S]
    b, t = ws.shape[:2]
    flat = ws.reshape(b * t, *ws.shape[2:])
    boxes = proposal_net.generate(flat).to(ws.device)            # [B*T,K,4]
    batch["proposals"] = boxes.reshape(b, t, cfg.num_proposals, 4)
    return batch
