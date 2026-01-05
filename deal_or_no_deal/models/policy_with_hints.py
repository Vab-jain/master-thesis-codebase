"""Actor-Critic policy that consumes base features and hint features.

Inputs are concatenated as `[base_flat, hint_feats]` where `hint_feats`
include 10 dims: act_onehot(5) + oA_norm(3) + confidence(1) + h_avail(1).

Heads:
- action logits: 5 (with masking for invalid actions)
- three oA categorical heads: each 5 logits (mask indices > oA_max[i])
- value head: scalar
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, List

import torch
import torch.nn as nn


class PolicyWithHints(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden, 5)
        self.oA_heads = nn.ModuleList([nn.Linear(hidden, 5) for _ in range(3)])
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(x)
        out = {
            "action_logits": self.action_head(h),  # [B,5]
            "oA_logits_0": self.oA_heads[0](h),   # [B,5]
            "oA_logits_1": self.oA_heads[1](h),
            "oA_logits_2": self.oA_heads[2](h),
            "value": self.value_head(h).squeeze(-1),  # [B]
        }
        return out

    @staticmethod
    def apply_masks(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Apply binary mask to logits: where mask==0 set to large negative."""
        if mask is None:
            return logits
        large_neg = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
        return torch.where(mask > 0, logits, large_neg)

    @staticmethod
    def apply_oa_caps(oalogits: torch.Tensor, oA_max: int) -> torch.Tensor:
        """Mask indices greater than oA_max by setting logits to large negative."""
        if oA_max >= 4:
            return oalogits
        if oalogits.dim() == 1:
            idx = torch.arange(5, device=oalogits.device)
            cap_mask = (idx <= int(oA_max)).to(oalogits.dtype)
        else:
            B = oalogits.shape[0]
            idx = torch.arange(5, device=oalogits.device).view(1, 5).expand(B, 5)
            cap_mask = (idx <= int(oA_max)).to(oalogits.dtype)
        return PolicyWithHints.apply_masks(oalogits, cap_mask)



