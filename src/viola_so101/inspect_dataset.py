"""Dataset inspector / SO-101 schema validator (entry point: ``viola-inspect``).

Prints ``robot_type``, fps, feature keys, and action/state dims for a
LeRobotDataset, and validates them against the VIOLA-SO101 config schema (top +
wrist cameras present, 6-DoF joint state/action). Uses
``LeRobotDatasetMetadata`` which downloads only the tiny ``meta/`` folder -- no
video, so validating a repo before pulling full data is cheap.

    viola-inspect --repo-id lerobot/svla_so100_stacking
"""

from __future__ import annotations

import argparse
import logging

from .config import VIOLAConfig
from .logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _shape_of(features: dict, key: str):
    feat = features.get(key)
    if feat is None:
        return None
    shape = feat.get("shape")
    return tuple(shape) if shape is not None else None


def inspect(repo_id: str, cfg: VIOLAConfig, root: str | None = None) -> bool:
    from lerobot.datasets import LeRobotDatasetMetadata  # [version seam]

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    features = meta.features

    print(f"\n=== {repo_id} ===")
    print(f"robot_type : {meta.robot_type}")
    print(f"fps        : {meta.fps}")
    print(f"episodes   : {meta.total_episodes}")
    print(f"frames     : {meta.total_frames}")
    print(f"cameras    : {list(meta.camera_keys)}")
    print("features:")
    for k, feat in features.items():
        names = feat.get("names")
        print(f"  {k:32s} dtype={feat.get('dtype'):8s} "
              f"shape={tuple(feat.get('shape', ()))} names={names}")

    # ---- validate against the SO-101 / VIOLA schema -----------------------
    ok = True
    checks: list[tuple[str, bool, str]] = []

    top_ok = cfg.workspace_image_key in meta.camera_keys
    checks.append((f"workspace camera '{cfg.workspace_image_key}'", top_ok,
                   "required"))
    wrist_ok = (cfg.wrist_image_key is None
                or cfg.wrist_image_key in meta.camera_keys)
    checks.append((f"wrist camera '{cfg.wrist_image_key}'", wrist_ok,
                   "required for dual-camera symmetric training"))

    state_shape = _shape_of(features, cfg.state_key)
    state_ok = state_shape == (cfg.state_dim,)
    checks.append((f"state '{cfg.state_key}' == ({cfg.state_dim},)", state_ok,
                   f"got {state_shape}"))
    action_shape = _shape_of(features, cfg.action_key)
    action_ok = action_shape == (cfg.action_dim,)
    checks.append((f"action '{cfg.action_key}' == ({cfg.action_dim},)", action_ok,
                   f"got {action_shape}"))

    fps_ok = meta.fps == cfg.fps
    checks.append((f"fps == {cfg.fps}", fps_ok, f"got {meta.fps}"))

    print("\nvalidation:")
    for name, passed, note in checks:
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}] {name}" + ("" if passed else f"  ({note})"))
        ok = ok and passed

    print(f"\n=> {'OK: matches SO-101 schema' if ok else 'MISMATCH (see FAIL above)'}\n")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect / validate a LeRobotDataset")
    ap.add_argument("--repo-id", required=True, nargs="+")
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    setup_logging(level="WARNING")
    cfg = VIOLAConfig.from_yaml(args.config) if args.config else VIOLAConfig()

    all_ok = True
    for repo_id in args.repo_id:
        all_ok = inspect(repo_id, cfg, root=args.root) and all_ok
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
