"""GMM action head (VIOLA Sec. 3.3 + Appendix A).

The action-token latent is passed through a two-layer MLP (1024 hidden each),
then projected to the parameters of a Gaussian Mixture Model with 5 modes over
the action space. Training loss is the negative log-likelihood of the
demonstrated action under the mixture.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MixtureSameFamily, Independent

from .config import VIOLAConfig


class GMMHead(nn.Module):
    def __init__(self, cfg: VIOLAConfig):
        super().__init__()
        self.cfg = cfg
        a, m = cfg.action_dim, cfg.n_modes
        self.mlp = nn.Sequential(
            nn.Linear(cfg.token_dim, cfg.mlp_hidden), nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(cfg.mlp_hidden, m * a)
        self.scale_head = nn.Linear(cfg.mlp_hidden, m * a)
        self.logit_head = nn.Linear(cfg.mlp_hidden, m)

    def _params(self, x: torch.Tensor):
        h = self.mlp(x)                                            # [B, hidden]
        b = x.shape[0]
        m, a = self.cfg.n_modes, self.cfg.action_dim
        means = self.mean_head(h).reshape(b, m, a)
        scales = F.softplus(self.scale_head(h)).reshape(b, m, a)
        scales = scales.clamp(self.cfg.min_std, self.cfg.max_std)
        logits = self.logit_head(h)                               # [B, m]
        return means, scales, logits

    def distribution(self, x: torch.Tensor) -> MixtureSameFamily:
        means, scales, logits = self._params(x)
        mixture = Categorical(logits=logits)
        components = Independent(Normal(means, scales), 1)        # over action dims
        return MixtureSameFamily(mixture, components)

    def loss(self, x: torch.Tensor, target_action: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(x)
        return -dist.log_prob(target_action).mean()

    @torch.no_grad()
    def act(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        means, scales, logits = self._params(x)
        if sample:
            return self.distribution(x).sample()
        # exploit: mean of the highest-weight mixture component
        best = logits.argmax(dim=-1)                              # [B]
        idx = best.view(-1, 1, 1).expand(-1, 1, means.shape[-1])
        return means.gather(1, idx).squeeze(1)                    # [B, action_dim]
