from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch

from env_wrappers.hint_injector import HintInjectorEnv
from train.common.rollout import make_base_env


class HintMode(str, Enum):
    none = "none"
    random = "random"
    llm = "llm"
    expert = "expert"


@dataclass
class HintAdapter:
    feature_dim: int

    def input_dim_from_obs(self, obs: Dict[str, Any], info: Dict[str, Any], hint_dim: int) -> int:
        # Compute from actual concatenation to avoid mismatch with presence/absence of hints
        sample = self.obs_to_input(obs, info)
        return int(sample.shape[0])

    def obs_to_input(self, obs: Dict[str, Any], info: Dict[str, Any]) -> np.ndarray:
        counts = np.array(obs["counts"], dtype=np.float32)
        my_utils = np.array(obs["my_utilities"], dtype=np.float32)
        partner_utils = np.array(obs.get("partner_utilities", [0, 0, 0]), dtype=np.float32)
        last_act = int(obs.get("last_partner_act", 4))
        last_act_oh = np.zeros(5, dtype=np.float32)
        last_act_oh[min(4, max(0, last_act))] = 1.0
        last_offer = np.array(obs.get("last_partner_offer_for_me", [0, 0, 0]), dtype=np.float32)
        tr = float(obs.get("turns_remaining", 0.0))
        base = np.concatenate([counts, my_utils, partner_utils, last_act_oh, last_offer, np.array([tr], dtype=np.float32)])
        hint = info.get("hint_features", {})
        act_onehot = np.array(hint.get("act_onehot", [0.0] * 5), dtype=np.float32)
        oA_norm = np.array(hint.get("oA_norm", [0.0] * 3), dtype=np.float32)
        confidence = np.array([hint.get("confidence", 0.0)], dtype=np.float32)
        h_avail = np.array([float(hint.get("h_avail", 0))], dtype=np.float32)
        return np.concatenate([base, act_onehot, oA_norm, confidence, h_avail])

    @staticmethod
    def obs_to_base_only(obs: Dict[str, Any]) -> np.ndarray:
        counts = np.array(obs["counts"], dtype=np.float32)
        my_utils = np.array(obs["my_utilities"], dtype=np.float32)
        partner_utils = np.array(obs.get("partner_utilities", [0, 0, 0]), dtype=np.float32)
        last_act = int(obs.get("last_partner_act", 4))
        last_act_oh = np.zeros(5, dtype=np.float32)
        last_act_oh[min(4, max(0, last_act))] = 1.0
        last_offer = np.array(obs.get("last_partner_offer_for_me", [0, 0, 0]), dtype=np.float32)
        tr = float(obs.get("turns_remaining", 0.0))
        return np.concatenate([counts, my_utils, partner_utils, last_act_oh, last_offer, np.array([tr], dtype=np.float32)])

    @staticmethod
    def get_action_mask(info: Dict[str, Any]) -> torch.Tensor:
        mask = np.array(info.get("action_mask", [1, 1, 1, 1, 1]), dtype=np.float32)
        return torch.from_numpy(mask)

    @staticmethod
    def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        large_neg = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
        if logits.dim() == 2:
            mask = mask.view(1, -1)
        return torch.where(mask > 0, logits, large_neg)

    @staticmethod
    def get_oa_caps(info: Dict[str, Any]) -> np.ndarray:
        return np.array(info.get("oA_max", [1, 1, 1]), dtype=np.int64)

    @staticmethod
    def apply_oa_cap(oalogits: torch.Tensor, oA_max: int) -> torch.Tensor:
        if oA_max >= 4:
            return oalogits
        idx = torch.arange(5, device=oalogits.device)
        cap_mask = (idx <= int(oA_max)).to(oalogits.dtype)
        return torch.where(cap_mask > 0, oalogits, torch.tensor(-1e9, dtype=oalogits.dtype, device=oalogits.device))


def make_env_with_hints(cfg: Dict[str, Any]) -> tuple[Callable[[], Any], Callable[[], Any] | None, HintAdapter]:
    max_turns = int(cfg.get("env", {}).get("max_turns", 10) or 10)
    mode = HintMode(cfg.get("hints", {}).get("mode", "none"))
    k = int(cfg.get("hints", {}).get("k", 1))
    prompt_path = str(cfg.get("hints", {}).get("prompt_path", "configs/llm_prompt.txt"))
    provider = str(cfg.get("hints", {}).get("provider", "local_hf"))
    expert_ckpt = cfg.get("hints", {}).get("expert_ckpt")

    def _env_factory():
        base = make_base_env(max_turns)
        if mode == HintMode.none:
            return base
        if mode == HintMode.random:
            return HintInjectorEnv(base, client=None, k=k, prompt_path=prompt_path, provider="random")
        if mode == HintMode.expert:
            # Allow None to trigger auto-discovery inside the wrapper
            return HintInjectorEnv(base, client=None, k=k, prompt_path=prompt_path, provider="expert", expert_ckpt=(str(expert_ckpt) if expert_ckpt else None))
        # LLM
        client = None
        try:
            from llm.client import GROQClient, LocalHFClient  # type: ignore
        except Exception:
            GROQClient = None  # type: ignore
            LocalHFClient = None  # type: ignore
        if provider.lower() == "groq" and GROQClient is not None:
            model_id = str(cfg.get("hints", {}).get("groq", {}).get("model", "llama-3.3-70b-versatile"))
            client = GROQClient(model=model_id)
        elif provider.lower() in ("local_hf", "local") and LocalHFClient is not None:
            model_id = str(cfg.get("hints", {}).get("local", {}).get("model", "llama3.1:latest"))
            client = LocalHFClient(model=model_id)
        return HintInjectorEnv(base, client=client, k=k, prompt_path=prompt_path, provider=provider)

    eval_factory = None
    adapter = HintAdapter(feature_dim=10)
    # annotate feature dimension for downstream calc
    cfg.setdefault("hints", {})["feature_dim"] = 10 if mode != HintMode.none else 0
    return _env_factory, eval_factory, adapter


