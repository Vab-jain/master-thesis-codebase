from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            # nn.Linear(hidden, hidden),
            # nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden, 5)
        self.oA_heads = nn.ModuleList([nn.Linear(hidden, 5) for _ in range(3)])
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {
            "action_logits": self.action_head(h),
            "oA_logits_0": self.oA_heads[0](h),
            "oA_logits_1": self.oA_heads[1](h),
            "oA_logits_2": self.oA_heads[2](h),
            "value": self.value_head(h).squeeze(-1),
        }


def build_policy(input_dim: int, hidden: int = 32) -> ActorCritic:
    return ActorCritic(input_dim=input_dim, hidden=hidden)


