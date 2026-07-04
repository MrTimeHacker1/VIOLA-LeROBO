# VIOLA-SO101

A faithful, from-scratch reimplementation of **VIOLA** (Zhu et al., *"VIOLA:
Imitation Learning for Vision-Based Manipulation with Object Proposal Priors"*,
CoRL 2022), adapted to the **SO-101** arm in the **LeRobot** ecosystem, with a
same-embodiment **pretrain → finetune** workflow.

The architecture stays faithful to the paper. The SO-101 adaptations are the
6-DoF joint action space and a torchvision Faster R-CNN RPN in place of Detic.
Training uses **Hugging Face Accelerate** (multi-GPU, bf16), structured
**logging**, **Weights & Biases**, and **Hugging Face Hub** checkpoints
(safetensors).

## Architecture at a glance

| | |
|---|---|
| Object proposals per frame `K` | 15 (frozen Faster R-CNN RPN, class-agnostic) |
| Temporal history `H` / frames | 9 / 10 |
| Workspace & wrist trunks | from-scratch ResNet-18, cut after `layer3` → 16×16 map |
| Region token | ROIAlign 6×6 → Linear → **+ box positional encoding** (added) |
| Context tokens/frame | 3: global (SpatialSoftmax), eye-in-hand (separate ResNet-18), proprio |
| Token dim `d` | 192 |
| Temporal token | `z_t = { h_{t-i} + PE_i }_{i=0..H}`, **index 0 = current frame** |
| Policy | encoder-only Transformer (4 layers, 6 heads, FFN 1024, post-norm) + action token |
| Head | 2-layer MLP → 5-mode Gaussian mixture; loss = mixture NLL |
| Sequence length | `1 + (H+1)*(K+3)` = **181**; ~11M parameters |

**Symmetric dual-camera.** Both stages use the identical token schema (workspace
+ wrist + proprio). The *only* intended differences between pretrain and finetune
are **data, learning rate, and which weights are frozen**.

## Install

```bash
git clone <this repo> && cd viola_so101_new
pip install -e .                 # core (model + training)
pip install -e ".[data,wandb]"   # + LeRobot dataset loading + W&B
```

### Environment notes

- **av1 video decoding.** The LeRobot SO-100 datasets are av1-encoded; torchcodec
  needs FFmpeg 7:
  ```bash
  conda install -c conda-forge "ffmpeg=7"
  python -c "from torchcodec.decoders import VideoDecoder"   # should not error
  ```
- **Output paths.** Caches and run directories default under
  `~/.cache/viola_so101/` (user-writable, outside the repo). Override with
  `--output-dir` / `--out` or `output_dir:` in a YAML config.
- **Hugging Face auth.** `hf auth login` (or set `HF_TOKEN`) for private
  datasets / `--push-to-hub`.

## Workflow

### 0. Validate a dataset against the SO-101 schema (cheap, metadata-only)

```bash
viola-inspect --repo-id lerobot/svla_so100_stacking lerobot/svla_so100_sorting \
              lerobot/svla_so100_pickplace
```
Prints `robot_type`, fps, feature keys, action/state dims and checks that the top
+ wrist cameras and 6-DoF joint schema are present.

### 1. (Optional) Precompute offline proposal caches

```bash
viola-preprocess --repo-id lerobot/svla_so100_stacking --out ~/viola_cache/stacking.npy
```
Writes a memmap `[N, 15, 4]` aligned 1:1 to frame order. If you skip this,
proposals are generated online on-device during training.

### 2. Pretrain (Stage 1) — pooled SO-100 datasets, full policy from scratch

```bash
accelerate config          # one-time; pick multi-GPU / bf16
accelerate launch -m viola_so101.train.pretrain \
    --config configs/pretrain.yaml \
    --repo-id lerobot/svla_so100_stacking \
             lerobot/svla_so100_sorting \
             lerobot/svla_so100_pickplace \
    --cache ~/viola_cache/stacking.npy \
            ~/viola_cache/sorting.npy \
            ~/viola_cache/pickplace.npy \
    --wandb
```
Each dataset is wrapped independently then concatenated (they differ on-disk:
v3.0 vs v2.1). Omit `--cache` to use online proposals. For a quick single-GPU
run drop `accelerate launch` env or use `accelerate launch --num_processes 1`.

Multi-GPU explicitly:
```bash
accelerate launch --multi_gpu --num_processes 4 -m viola_so101.train.pretrain ...
```

### 3. Finetune (Stage 2) — your ~50 SO-101 demos

```bash
accelerate launch -m viola_so101.train.finetune \
    --config configs/finetune.yaml \
    --repo-id <your/so101_demos> \
    --pretrained ~/.cache/viola_so101/runs/pretrain/best \
    --wandb --push-to-hub <you>/viola-so101
```
Loads the Stage-1 checkpoint non-strictly (reports missing/unexpected keys),
freezes both ResNet-18 backbones (only those present in the checkpoint), and
trains the rest at `lr=1e-5`. Online proposals by default.

### 4. Deploy

```bash
viola-deploy --checkpoint ~/.cache/viola_so101/runs/finetune/best --port /dev/ttyACM0
```
Runs a sliding window of the last `H+1` observations at `fps/frame_stride` Hz,
runs the RPN on the top image, and emits the highest-weight GMM mean as a joint
command. The LeRobot robot-I/O calls are marked `[robot seam]` in
`src/viola_so101/deploy.py` — adapt them to your installed LeRobot version.

## Pretraining datasets

| repo_id | version | episodes | frames |
|---|---|---|---|
| `lerobot/svla_so100_stacking`  | v3.0 | 56 | 22,956 |
| `lerobot/svla_so100_sorting`   | v3.0 | 52 | 35,713 |
| `lerobot/svla_so100_pickplace` | v2.1 | 50 | 19,631 |

All share the 6-DoF joint schema (`main_shoulder_pan, main_shoulder_lift,
main_elbow_flex, main_wrist_flex, main_wrist_roll, main_gripper`) and both
cameras `observation.images.top` / `observation.images.wrist`. Validate with
`viola-inspect` before training.

## Tests

```bash
pip install -e ".[dev]"
pytest -q                                    # smoke + dry-run, CPU, no network
# zero deprecation/future warnings on the model + augmentation path:
python -W error::DeprecationWarning -W error::FutureWarning tests/test_smoke.py
```

## Package layout

```
src/viola_so101/
  config.py            positional_encoding.py   region_encoder.py
  context_encoder.py   transformer_policy.py    gmm_head.py
  policy.py            proposals.py             logging_utils.py
  data/{dataset,augmentation}.py
  train/{loop,pretrain,finetune}.py
  preprocess.py  deploy.py  inspect_dataset.py
```

## Notes on faithfulness / design choices

- `token_dim = 192` and Transformer `dropout = 0.1` are design choices (the paper
  fixes neither); 192 is divisible by 4 (4-corner box PE) and 6 (heads).
- Box PE and temporal PE are **added** (not concatenated), keeping tokens at `d`.
- Proposals come from torchvision's Faster R-CNN RPN (frozen, no-grad, never a
  trainable parameter) in place of the paper's Detic.
