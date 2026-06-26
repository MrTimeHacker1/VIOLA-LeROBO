"""Real-time deployment on the SO-101 (VIOLA Sec. 3.3, Fig. 2).

Maintains a sliding window of the last H+1 observations and, each tick, runs the
frozen proposal net + policy to produce one joint-space action. The control loop
should run at native_fps / frame_stride Hz to match the temporal spacing used in
training (see README).

NOTE [robot seam]: the camera reads and the joint command write go through the
LeRobot robot interface. The exact calls depend on your LeRobot version; the two
clearly marked spots below are all you need to adapt.
"""

from __future__ import annotations
from collections import deque
import torch
from torchvision.transforms import v2

from viola.config import VIOLAConfig
from viola.policy import VIOLAPolicy
from viola.proposals import ProposalNetwork


def _resize(img, size):
    return v2.functional.resize(img, [size, size], antialias=True)


class ViolaController:
    def __init__(self, checkpoint: str, device: str = "cuda"):
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.cfg = VIOLAConfig(**{k: v for k, v in ckpt["cfg"].items()
                                  if k in VIOLAConfig().__dict__})
        self.device = device
        self.policy = VIOLAPolicy(self.cfg).to(device).eval()
        self.policy.load_state_dict(ckpt["model"])
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
                self._ws.append(ws); self._wr.append(wr); self._st.append(st)

    @torch.no_grad()
    def act(self, workspace_img, wrist_img, state, sample: bool = False):
        """workspace_img / wrist_img: float CHW in [0,1]; state: [state_dim]."""
        cfg = self.cfg
        ws = _resize(workspace_img, cfg.image_size).to(self.device)
        wr = (_resize(wrist_img, cfg.wrist_image_size).to(self.device)
              if cfg.wrist_image_key is not None else None)
        st = state.float().to(self.device)

        self._prime(ws, wr, st)
        self._ws.append(ws); self._st.append(st)
        if wr is not None:
            self._wr.append(wr)

        ws_seq = torch.stack(list(self._ws), 0).unsqueeze(0)      # [1,T,3,S,S]
        st_seq = torch.stack(list(self._st), 0).unsqueeze(0)      # [1,T,state]
        boxes = self.proposals.generate(ws_seq[0]).to(self.device).unsqueeze(0)

        batch = {"workspace_images": ws_seq, "states": st_seq, "proposals": boxes}
        if cfg.wrist_image_key is not None:
            batch["wrist_images"] = torch.stack(list(self._wr), 0).unsqueeze(0)

        action = self.policy.select_action(batch, sample=sample)  # [1, action_dim]
        return action.squeeze(0).cpu()


def run(checkpoint: str, device: str = "cuda"):
    """Closed-loop control sketch. Adapt the two [robot seam] spots."""
    from lerobot.robots.so101_follower import SO101Follower  # [robot seam]

    ctrl = ViolaController(checkpoint, device=device)
    robot = SO101Follower(...)                                   # [robot seam] configure ports/cameras
    robot.connect()
    ctrl.reset()
    try:
        while True:
            obs = robot.get_observation()                        # [robot seam]
            ws = obs[ctrl.cfg.workspace_image_key]
            wr = obs.get(ctrl.cfg.wrist_image_key)
            st = obs[ctrl.cfg.state_key]
            action = ctrl.act(ws, wr, st)
            robot.send_action(action)                            # [robot seam]
    finally:
        robot.disconnect()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run(args.checkpoint, args.device)
