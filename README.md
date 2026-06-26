# VIOLA for SO-101 (LeRobot)

A faithful, from-scratch reimplementation of **VIOLA** — *Imitation Learning for
Vision-Based Manipulation with Object Proposal Priors* (Zhu et al., CoRL 2022) —
wired into the **LeRobot / SO-101** ecosystem with a **pretrain → fine-tune**
workflow.

The model architecture follows the paper exactly. The only deliberate
adaptations are the things the paper's single-task Franka setup didn't need:
joint-space actions instead of OSC end-effector control, a torchvision RPN in
place of Detic for proposals, and a Stage-1 policy-pretraining wrapper around the
otherwise-unchanged architecture.

---

## What maps to what (paper → this code)

| Paper component | File | Notes |
|---|---|---|
| General object proposals (frozen RPN) | `viola/proposals.py` | torchvision Faster R-CNN RPN, class-agnostic, always frozen / `no_grad` |
| Region features (ResNet-18 spatial map, from scratch, + ROIAlign + box PE) | `viola/region_encoder.py` | 256→16×16 map, 6×6 ROIAlign, visual **+** positional feature |
| Context features (global Spatial-Softmax, eye-in-hand ResNet-18, proprio) | `viola/context_encoder.py` | wrist trunk has its own weights |
| Box & temporal positional encodings | `viola/positional_encoding.py` | Appendix A formulas, base frequency 10 |
| Temporal composition (H+1 per-step features + temporal PE) | `viola/policy.py` | window ordering oldest→current |
| Transformer policy + action token | `viola/transformer_policy.py` | 4 layers, 6 heads, FFN 1024, post-norm |
| GMM head (5 modes, NLL loss) | `viola/gmm_head.py` | `MixtureSameFamily(Categorical, Independent(Normal))` |
| Augmentation (color jitter / pixel shift / random erasing / box noise) | `data/augmentation.py` | paper parameters in Appendix A |

Hard constants from the paper (K=15, H=9, 16×16 map, 6×6 ROIAlign, 4 layers,
6 heads, FFN 1024, 5 GMM modes, AdamW, lr 1e-4, cosine annealing, 50 epochs,
batch 16, grad-clip 0.1) all live in `viola/config.py`.

---

## Your cameras

VIOLA uses two cameras, which map to your SO-101 as:

- **whole-state camera** (external, sees the full workspace) → VIOLA **workspace
  camera**. Proposals run on this image; it feeds the region tokens and the
  global context token. Config key `observation.images.top`.
- **eye-piece camera** (on the wrist/gripper) → VIOLA **eye-in-hand camera**.
  No proposals; it feeds the wrist context token to resolve occlusions. Config
  key `observation.images.wrist`.

Set these keys in `viola/config.py` (or the YAMLs) to match your dataset's
feature names. Single-camera setups can pass `--no-wrist`.

---

## Install

```bash
pip install -r requirements.txt
```

`torch`, `torchvision`, `numpy`, and `lerobot` are required. The first run of the
proposal network downloads Faster R-CNN weights from torchvision.

---

## The three stages

### Stage 0 — proposals (always frozen)
The proposal network is never trained, at any stage. It is not part of the
policy's parameters.

### Stage 1 — pretrain on pooled SO-100/101 data
Trains the **full policy from random init** on community datasets that share the
SO-101 6-DoF joint schema. Same-embodiment pretraining is what lets the whole
policy — GMM head included — transfer to your task.

Recommended datasets: `lerobot/svla_so100_stacking`,
`lerobot/svla_so101_pickplace`.

```bash
# 1) precompute proposals once per dataset (offline cache)
python scripts/preprocess.py --repo-id lerobot/svla_so100_stacking \
    --out caches/svla_so100_stacking.npy --device cuda
python scripts/preprocess.py --repo-id lerobot/svla_so101_pickplace \
    --out caches/svla_so101_pickplace.npy --device cuda

# 2) pretrain
python scripts/pretrain.py \
    --repo-id lerobot/svla_so100_stacking lerobot/svla_so101_pickplace \
    --cache  caches/svla_so100_stacking.npy caches/svla_so101_pickplace.npy \
    --out runs/pretrain --device cuda
```

### Stage 2 — fine-tune on your stacking episodes
Record ~50 teleoperated stacking demos with `lerobot-record` (both cameras, each
episode covering the **full** multi-step horizon — see "Stacking data" below).
Then:

```bash
python scripts/finetune.py \
    --repo-id <your-hf-user>/so101_stacking \
    --pretrained runs/pretrain/best.pt \
    --out runs/finetune --device cuda
```

By default both ResNet-18 backbones are frozen and only the reasoning/head
weights adapt, at `lr = 1e-5`. Pass `--no-freeze` to fine-tune everything, or
call `policy.unfreeze_backbones()` if the loss plateaus.

### Deploy
```bash
python deploy/inference.py --checkpoint runs/finetune/best.pt --device cuda
```

---

## Online vs offline proposals

- **Offline** (Stage 1): proposals are computed once by `scripts/preprocess.py`
  and cached as a memmap `[N, K, 4]` aligned 1:1 with the dataset's frames.
  Training loads boxes from disk. Use this for the large community datasets.
- **Online** (Stage 2 + deployment): proposals are generated live on the GPU
  inside the loop (`data/dataset.add_online_proposals`). Fine for the small
  fine-tune set and mandatory on the real robot.

---

## Control rate / `frame_stride`

The paper runs at 20 Hz with H=9 (a 0.45 s window). LeRobot community data is
typically recorded at 30 fps. `frame_stride` selects the temporal spacing of the
H+1 window: effective rate ≈ `native_fps / frame_stride`. Default is `1`
(consecutive frames). **Run your deployment control loop at the same effective
rate you trained with**, so the temporal spacing the policy sees at test time
matches training.

---

## Stacking data note

"Long-horizon stacking" means the policy must learn inter-step transitions
(grasp A → place → grasp B → place on A …). Your fine-tune episodes must each
span the **whole** task, not isolated sub-grasps, or the temporal composition
has nothing to learn the transitions from.

---

## Version seams to verify

A few spots depend on your installed LeRobot / robot version and are clearly
marked in-code:

- `data/dataset.py` — `lerobot.datasets.lerobot_dataset.LeRobotDataset` import
  and the `episode_index` metadata column used by `_episode_starts`.
- `deploy/inference.py` — the three `[robot seam]` calls
  (`get_observation` / `send_action` / robot construction).

The model, augmentation, training loop, and proposal code have no such
dependency and are unit-smoke-tested in `tests/smoke_test.py`:

```bash
PYTHONPATH=. python tests/smoke_test.py
```

---

## Faithful vs adapted

**Faithful to the paper:** all architecture dimensions and counts; from-scratch
ResNet-18 spatial map (Appendix B.2); visual **+** positional region feature;
three context tokens; temporal composition with temporal PE; encoder-only
transformer with a learnable action token; 5-mode GMM with NLL; AdamW + cosine
annealing + grad-clip 0.1; the full augmentation suite; lowest-train-loss
checkpointing.

**Adapted for SO-101 (marked `[SO-101]` in code):** 6-DoF joint-space action/
state instead of OSC end-effector; torchvision RPN instead of Detic (same
"frozen pretrained RPN" role); and the Stage-1 policy-pretraining wrapper, which
is an addition on top of — not a change to — the paper's architecture.
