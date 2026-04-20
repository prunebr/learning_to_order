from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


@dataclass
class ReinforceConfig:
    lr: float = 3e-4
    entropy_beta: float = 1e-3
    grad_clip: float = 1.0
    baseline_ema: float = 0.95  # EMA do baseline


class ReinforceTrainer:
    def __init__(self, policy: nn.Module, cfg: ReinforceConfig, device: str = "cpu"):
        self.policy = policy.to(device)
        self.cfg = cfg
        self.device = torch.device(device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

        self._baseline: Optional[torch.Tensor] = None

    def update(self, reward: torch.Tensor, logprob_sum: torch.Tensor, entropy_sum: torch.Tensor) -> Dict[str, float]:
        """
        reward: escalar
        logprob_sum: soma de log-probs do episódio
        entropy_sum: soma de entropias do episódio
        """
        reward = reward.to(self.device)
        logprob_sum = logprob_sum.to(self.device)
        entropy_sum = entropy_sum.to(self.device)

        if self._baseline is None:
            self._baseline = reward.detach()
        else:
            self._baseline = self.cfg.baseline_ema * self._baseline + (1.0 - self.cfg.baseline_ema) * reward.detach()

        adv = (reward - self._baseline).detach()

        loss = -(adv * logprob_sum) - (self.cfg.entropy_beta * entropy_sum)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)

        self.opt.step()

        return {
            "loss": float(loss.item()),
            "reward": float(reward.item()),
            "baseline": float(self._baseline.item()),
            "adv": float(adv.item()),
            "logprob_sum": float(logprob_sum.item()),
            "entropy_sum": float(entropy_sum.item()),
        }
