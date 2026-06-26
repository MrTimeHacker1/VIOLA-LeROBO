"""Configuration for the VIOLA-style object-centric policy.

From-scratch reimplementation of:
    Zhu et al., "VIOLA: Imitation Learning for Vision-Based Manipulation with
    Object Proposal Priors", CoRL 2022.

Every architectural constant is taken directly from the paper (main text +
Appendix A). Additions specific to the SO-101 / LeRobot setting and to the
pretrain -> fine-tune workflow are marked "[SO-101]".

Hard constants from the paper:
    K (real robot)           = 15        (Sec. 3.3 Implementation Details)
    H                        = 9         -> H+1 = 10 frames (Sec. 3.3)
    control rate             = 20 Hz     (Fig. 2 / Appendix C)
    spatial feature map      = 16 x 16   (Appendix A)
    ROIAlign output          = 6 x 6     (Appendix A)
    transformer layers       = 4         (Appendix A)
    attention heads          = 6         (Appendix A)
    FFN hidden               = 1024      (Appendix A)
    GMM modes                = 5         (Appendix A)
    MLP hidden (head)        = 1024      (Appendix A)
    optimizer                = AdamW     (Appendix A)
    lr                       = 1e-4      (Appendix A)
    scheduler                = cosine annealing (Appendix A)
    epochs                   = 50        (Appendix A)
    batch size               = 16        (Appendix A)
    grad clip (long-horizon) = 0.1       (Appendix A; stacking is long-horizon)
    PE base frequency        = 10        (Appendix A)
    color jitter             = b/c/s 0.3, hue 0.05, on 90% (Appendix A)
    pixel shift              = 4 px      (Appendix A)
    random erasing           = p=0.5, scale (0.02,0.05), ratio (0.5,1.5) (App. A)
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class VIOLAConfig:
    # ---------------- observation keys (LeRobotDataset feature names) [SO-101] -
    workspace_image_key: str = "observation.images.top"     # whole-state camera
    wrist_image_key: str | None = "observation.images.wrist"  # eye-in-hand camera
    state_key: str = "observation.state"
    action_key: str = "action"

    # ---------------- image sizes -------------------------------------------
    # 256x256 workspace input through ResNet-18 (stride 16) yields the paper's
    # exact 16x16 spatial feature map.
    image_size: int = 256
    wrist_image_size: int = 256

    # ---------------- object proposals --------------------------------------
    num_proposals: int = 15          # K, real-robot value from the paper
    proposals_key: str = "proposals"  # boxes are normalized xyxy in [0,1]

    # ---------------- region / spatial features -----------------------------
    spatial_channels: int = 256      # channels of ResNet-18 layer3 output
    spatial_map_size: int = 16       # 16x16 feature map (paper)
    roi_output_size: int = 6         # 6x6 ROIAlign (paper)
    backbone_stride: int = 16        # input/feature-map stride for 256 -> 16

    # ---------------- token dimensionality ----------------------------------
    # 192 is divisible by 4 (box positional encoding uses 4 corner coords) and
    # by n_heads=6 (32 dims per head).
    token_dim: int = 192

    # ---------------- temporal composition ----------------------------------
    history: int = 9                 # H. Policy sees the last H+1 frames.
    # Frame stride used to build the temporal window from the recorded data.
    # Set so that (native_fps / frame_stride) ~ control rate. See README. [SO-101]
    frame_stride: int = 1

    # ---------------- transformer policy ------------------------------------
    n_layers: int = 4
    n_heads: int = 6
    ffn_dim: int = 1024
    dropout: float = 0.1

    # ---------------- GMM action head ---------------------------------------
    action_dim: int = 6              # SO-101 joint space [SO-101]
    state_dim: int = 6               # SO-101 proprioception [SO-101]
    n_modes: int = 5
    mlp_hidden: int = 1024
    min_std: float = 1e-4
    max_std: float = 10.0

    # ---------------- positional-encoding base frequency --------------------
    pe_base: float = 10.0            # paper uses 10 (short sequences)

    # ---------------- optimisation ------------------------------------------
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 16
    grad_clip: float = 0.1           # 0.1 for long-horizon tasks (paper)

    # ---------------- data augmentation -------------------------------------
    color_jitter_prob: float = 0.9   # applied to 90% of samples
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3
    color_jitter_saturation: float = 0.3
    color_jitter_hue: float = 0.05
    pixel_shift: int = 4             # max pixel translation
    random_erase_prob: float = 0.5
    random_erase_scale: tuple[float, float] = (0.02, 0.05)
    random_erase_ratio: tuple[float, float] = (0.5, 1.5)
    box_noise_std: float = 0.01      # small gaussian noise on proposal coords

    # ---------------- fine-tuning behaviour [SO-101] ------------------------
    finetune_lr: float = 1e-5        # 10x lower than pretrain
    freeze_backbones_on_finetune: bool = True

    # ------------------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return self.history + 1

    @property
    def n_context_tokens(self) -> int:
        # global + proprioception (+ wrist if present)
        return 2 + (1 if self.wrist_image_key is not None else 0)

    @property
    def tokens_per_frame(self) -> int:
        return self.num_proposals + self.n_context_tokens

    @property
    def seq_len(self) -> int:
        # action token + all observation tokens
        return 1 + self.n_frames * self.tokens_per_frame

    def __post_init__(self):
        assert self.token_dim % 4 == 0, "token_dim must be divisible by 4 (box PE)"
        assert self.token_dim % self.n_heads == 0, "token_dim must divide by n_heads"

    # ---- stage factories ---------------------------------------------------
    @classmethod
    def for_pretrain(cls, **overrides) -> "VIOLAConfig":
        cfg = cls(**overrides)
        cfg.lr = cls.lr
        return cfg

    @classmethod
    def for_finetune(cls, **overrides) -> "VIOLAConfig":
        cfg = cls(**overrides)
        cfg.lr = cfg.finetune_lr
        return cfg
