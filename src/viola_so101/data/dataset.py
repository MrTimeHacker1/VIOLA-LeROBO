"""LeRobot dataset wrapper for VIOLA (temporal windows + proposals).

Wraps a ``LeRobotDataset`` and, for each center frame, builds the H+1 frame
window the policy consumes. Windows are constructed by explicit global indexing
(NOT ``delta_timestamps``) so that offline proposal caches align 1:1 with frames.

Frame ordering in the returned window (paper-faithful):
    index 0 = CURRENT frame (t), index H = OLDEST frame (t - H*stride).
Windows are clamped at the start of an episode by repeating the episode's first
frame (padding). The demonstrated target is the action at the current frame.

NOTE [version seam]: LeRobot 0.5.1 flat namespace
(``lerobot.datasets.LeRobotDataset``) and the ``episode_index`` column. If your
installed version differs, adjust ``_load_lerobot_dataset`` and
``_episode_starts`` only.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import v2

from ..config import VIOLAConfig
from .augmentation import VIOLAAugmentation

logger = logging.getLogger(__name__)


def _load_lerobot_dataset(repo_id: str, root: str | None = None):
    # Lazy import so the model/training utilities don't require lerobot. [seam]
    from lerobot.datasets import LeRobotDataset
    return LeRobotDataset(repo_id, root=root)


def _to_float_chw(img: torch.Tensor) -> torch.Tensor:
    # LeRobot returns CHW float32 [0,1]; handle a uint8 stream defensively.
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    return img.float()


def _episode_starts(ds) -> np.ndarray:
    """Return, for every global frame, the global start index of its episode."""
    ep = np.asarray(ds.hf_dataset["episode_index"])
    counts = np.bincount(ep)
    starts = np.cumsum(counts) - counts          # start index per episode id
    return starts[ep]                            # [N]


class VIOLALeRobotDataset(Dataset):
    def __init__(self, repo_id_or_dataset, cfg: VIOLAConfig, train: bool = True,
                 proposal_cache_path: str | None = None, root: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.train = train
        if isinstance(repo_id_or_dataset, str):
            self.ds = _load_lerobot_dataset(repo_id_or_dataset, root=root)
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
            assert self.proposals.shape[0] == self.n, (
                f"proposal cache misaligned: {self.proposals.shape[0]} != {self.n}"
            )
            assert self.proposals.shape[1] == cfg.num_proposals, "K mismatch"

    def __len__(self):
        return self.n

    @property
    def has_proposals(self) -> bool:
        return self.proposals is not None

    def _window_indices(self, center: int) -> list[int]:
        """index 0 = current (center), index H = oldest, clamped to episode start."""
        s = int(self.ep_start[center])
        st = self.cfg.frame_stride
        H = self.cfg.history
        return [max(s, center - i * st) for i in range(H + 1)]

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
            ws_params = self.aug.sample_params()
            ws = self.aug.apply_image(ws, ws_params, erase=True)
            if wr is not None:
                # independent sensor -> its own shared-across-time params
                wr = self.aug.apply_image(wr, self.aug.sample_params(), erase=False)
            if self.proposals is not None:
                # keep cached boxes consistent with the pixel shift, then noise
                boxes = self.aug.apply_boxes(boxes, ws_params, shift=True)

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


def build_pooled_dataset(repo_ids, cfg: VIOLAConfig, caches=None,
                         train: bool = True, roots=None) -> ConcatDataset:
    """Wrap each repo INDEPENDENTLY (they may differ on-disk, e.g. v3.0 vs v2.1)
    then concatenate. ``caches`` / ``roots`` are optional lists aligned with
    ``repo_ids`` (use None entries for repos without a cache)."""
    caches = caches or [None] * len(repo_ids)
    roots = roots or [None] * len(repo_ids)
    assert len(caches) == len(repo_ids), "caches must align with repo_ids"
    parts = []
    for repo_id, cache, root in zip(repo_ids, caches, roots):
        d = VIOLALeRobotDataset(repo_id, cfg, train=train,
                                proposal_cache_path=cache, root=root)
        logger.info("Loaded %s: %d frames (cache=%s).", repo_id, len(d),
                    "yes" if cache else "online")
        parts.append(d)
    return ConcatDataset(parts)


@torch.no_grad()
def add_online_proposals(batch: dict, proposal_net, cfg: VIOLAConfig,
                         train: bool = False, aug: VIOLAAugmentation | None = None
                         ) -> dict:
    """Fill ``batch['proposals']`` by running the frozen proposal net on-device.

    Used when no offline cache exists (e.g. fine-tuning). The detector runs on
    the (already-augmented) workspace frames, so boxes are consistent with the
    pixels by construction. Box noise is added when training, to match the
    offline path's regularisation.
    """
    ws = batch["workspace_images"]                                # [B,T,3,S,S]
    b, t = ws.shape[:2]
    flat = ws.reshape(b * t, *ws.shape[2:])
    boxes = proposal_net.generate(flat).to(ws.device)            # [B*T,K,4]
    boxes = boxes.reshape(b, t, cfg.num_proposals, 4)
    if train and cfg.box_noise_std > 0:
        noise = torch.randn_like(boxes) * cfg.box_noise_std
        boxes = (boxes + noise).clamp(0.0, 1.0)
    batch["proposals"] = boxes
    return batch
