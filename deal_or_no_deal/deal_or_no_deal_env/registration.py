from __future__ import annotations

from typing import Any, Dict, Optional


def _get_gym_api():
    try:
        import gymnasium as gym  # type: ignore
        return gym
    except Exception:
        import gym  # type: ignore
        return gym


def register_deal_or_no_deal(
    env_id: str = "DealOrNoDialog-v0",
    entry_point: str = "deal_or_no_deal_env.env:NegotiationEnv",
    kwargs: Optional[Dict[str, Any]] = None,
    max_episode_steps: Optional[int] = None,
) -> str:
    """Register the environment with Gymnasium/Gym and return the env_id.

    If an env with the same id exists, it is overwritten.
    """

    gym = _get_gym_api()
    try:
        if env_id in gym.registry:
            # gymnasium has registry as a dict-like; older gym may differ
            gym.registry.pop(env_id, None)
    except Exception:
        pass

    gym.register(id=env_id, entry_point=entry_point, kwargs=kwargs or {}, max_episode_steps=max_episode_steps)
    return env_id


