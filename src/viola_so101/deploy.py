"""Real-time deployment on the SO-101 (VIOLA Sec. 3.3, Fig. 2).

Maintains a sliding window of the last H+1 observations and, each tick, runs the
frozen proposal net (on the top image only) + policy to produce one joint-space
action (the mean of the highest-weight GMM component). The window is delivered
to the policy with index 0 = CURRENT (newest) frame, matching training. The
control loop runs at fps / frame_stride Hz to match the training spacing.

NOTE [robot seam]: the LeRobot robot I/O differs by version. In lerobot 0.5.1,
``SO101Follower.get_observation()`` / ``send_action()`` speak ``{motor}.pos``
dicts (not ``observation.state`` tensors) plus raw HWC camera arrays. The three
clearly-marked spots below (robot construction, obs read, action write) are all
you need to adapt; ``_obs_to_inputs`` / ``_action_to_command`` are the seam
converters between the robot's dict format and the policy's tensor format.
"""

from __future__ import annotations

import argparse
import logging
from collections import deque

import torch
from torchvision.transforms import v2

from .config import VIOLAConfig
from .policy import VIOLAPolicy
from .proposals import ProposalNetwork

logger = logging.getLogger(__name__)

# SO-101 joint order (state/action dimension order used by the datasets).
SO101_MOTORS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def _resize(img, size):
    return v2.functional.resize(img, [size, size], antialias=True)


class ViolaController:
    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        self.device = device
        self.policy = VIOLAPolicy.from_pretrained(checkpoint_dir).to(device).eval()
        self.cfg = self.policy.cfg
        self.proposals = ProposalNetwork(self.cfg.num_proposals, device=device)
        self.reset()

    def reset(self):
        self._ws = deque(maxlen=self.cfg.n_frames)
        self._wr = deque(maxlen=self.cfg.n_frames)
        self._st = deque(maxlen=self.cfg.n_frames)

    def _prime(self, ws, wr, st):
        # On the first tick, fill the whole window with the current observation.
        if len(self._ws) == 0:
            for _ in range(self.cfg.n_frames):
                self._ws.append(ws)
                self._wr.append(wr)
                self._st.append(st)

    @torch.no_grad()
    def act(self, workspace_img, wrist_img, state, sample: bool = False):
        """workspace_img / wrist_img: float CHW in [0,1]; state: [state_dim].
        Returns the commanded joint action as a [state_dim] CPU tensor."""
        cfg = self.cfg
        ws = _resize(workspace_img, cfg.image_size).to(self.device)
        wr = (_resize(wrist_img, cfg.wrist_image_size).to(self.device)
              if cfg.wrist_image_key is not None else None)
        st = state.float().to(self.device)

        self._prime(ws, wr, st)
        self._ws.append(ws)
        self._st.append(st)
        if wr is not None:
            self._wr.append(wr)

        # index 0 = current (newest); deque appends newest on the right.
        ws_seq = torch.stack(list(self._ws)[::-1], 0).unsqueeze(0)   # [1,T,3,S,S]
        st_seq = torch.stack(list(self._st)[::-1], 0).unsqueeze(0)   # [1,T,state]
        boxes = self.proposals.generate(ws_seq[0]).to(self.device).unsqueeze(0)

        batch = {"workspace_images": ws_seq, "states": st_seq, "proposals": boxes}
        if cfg.wrist_image_key is not None:
            batch["wrist_images"] = torch.stack(list(self._wr)[::-1], 0).unsqueeze(0)

        action = self.policy.select_action(batch, sample=sample)     # [1, A]
        return action.squeeze(0).cpu()


# ---- seam converters between LeRobot dicts and policy tensors --------------
def _camera(obs: dict, feature_key: str):
    """Fetch a camera frame from the robot obs dict by the trained feature key.

    The dataset feature is e.g. ``observation.images.top``; the robot obs dict
    keys the same camera by its short config name (``top``). Try both.
    """
    short = feature_key.split(".")[-1]
    return obs[short] if short in obs else obs[feature_key]


def _obs_to_inputs(obs: dict, cfg: VIOLAConfig):
    """[robot seam] Convert a lerobot observation dict to policy inputs.

    In lerobot 0.5.1 ``obs`` has ``{motor}.pos`` floats and per-camera HWC
    uint8/np arrays. Adapt the camera key names to your robot config.
    """
    import numpy as np

    def to_chw(img):
        t = torch.as_tensor(np.asarray(img))
        if t.ndim == 3 and t.shape[-1] == 3:          # HWC -> CHW
            t = t.permute(2, 0, 1)
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        return t.float()

    ws = to_chw(_camera(obs, cfg.workspace_image_key))
    wr = (to_chw(_camera(obs, cfg.wrist_image_key))
          if cfg.wrist_image_key is not None else None)
    state = torch.tensor([obs[f"{m}.pos"] for m in SO101_MOTORS], dtype=torch.float32)
    return ws, wr, state


def _action_to_command(action: torch.Tensor) -> dict:
    """[robot seam] Convert a policy action tensor to a lerobot action dict."""
    a = action.tolist()
    return {f"{m}.pos": float(v) for m, v in zip(SO101_MOTORS, a)}


def run(checkpoint_dir: str, device: str = "cuda", port: str = "/dev/ttyACM0"):
    """Closed-loop control sketch. Adapt the three [robot seam] spots."""
    import time

    from lerobot.robots.so_follower import (  # [robot seam] lerobot 0.5.1
        SO101Follower, SO101FollowerConfig,
    )

    ctrl = ViolaController(checkpoint_dir, device=device)
    # [robot seam] configure your port + cameras (must match the trained keys).
    robot = SO101Follower(SO101FollowerConfig(port=port))
    robot.connect()
    ctrl.reset()
    period = 1.0 / ctrl.cfg.control_hz
    try:
        while True:
            t0 = time.perf_counter()
            obs = robot.get_observation()                    # [robot seam]
            ws, wr, st = _obs_to_inputs(obs, ctrl.cfg)
            action = ctrl.act(ws, wr, st)
            robot.send_action(_action_to_command(action))    # [robot seam]
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        robot.disconnect()


def main() -> None:
    ap = argparse.ArgumentParser(description="Deploy a VIOLA-SO101 policy")
    ap.add_argument("--checkpoint", required=True,
                    help="checkpoint dir (contains model.safetensors + config.json)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--port", default="/dev/ttyACM0")
    args = ap.parse_args()
    run(args.checkpoint, device=args.device, port=args.port)


if __name__ == "__main__":
    main()
