from __future__ import annotations

from typing import Any, Callable, Dict

import gymnasium as gym

from deal_or_no_deal_env import register_deal_or_no_deal
from deal_or_no_deal_env.env import NegotiationConfig


def make_base_env(max_turns: int) -> Any:
    env_id = register_deal_or_no_deal()
    return gym.make(env_id, config=NegotiationConfig(max_turns=max_turns, use_dataset=False))


