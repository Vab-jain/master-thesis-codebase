from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _get_gym_api():
    try:
        import gymnasium as gym  # type: ignore
        return gym
    except Exception:
        import gym  # type: ignore
        return gym


gym = _get_gym_api()


@dataclass
class NegotiationConfig:
    # Item counts i ∈ {1..4}^3, utilities u ∈ {0..10}^3
    min_count: int = 0  # allow 0 for flexibility with dataset outputs
    max_count: int = 4
    min_utility: int = 0
    max_utility: int = 10
    max_turns: int = 10  # total turns (both agents combined)
    agent_starts: Optional[bool] = None  # None=random
    # Dataset integration
    use_dataset: bool = True
    dataset_script_path: Optional[str] = None  # path to deal_or_no_dialog.py
    dataset_config_name: str = "self_play"  # or "dialogues"
    # Observation privacy
    reveal_partner_utilities: bool = False
    # Optional normalization to enforce max points budget per the paper
    normalize_utilities_to_max_points: bool = False
    max_points_budget: int = 10


class NegotiationEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 30}

    ACT_PROPOSE = 0
    ACT_INSIST = 1
    ACT_AGREE = 2
    ACT_DISAGREE = 3
    ACT_END = 4

    def __init__(self, config: Optional[NegotiationConfig] = None):
        super().__init__()
        self.config = config or NegotiationConfig()
        self.rng = np.random.default_rng()
        self._datasets_loaded = False
        self._dataset = None

        # Observation space
        self.observation_space = gym.spaces.Dict(
            {
                "counts": gym.spaces.MultiDiscrete([self.config.max_count + 1] * 3),
                "my_utilities": gym.spaces.MultiDiscrete([self.config.max_utility + 1] * 3),
                "partner_utilities": gym.spaces.MultiDiscrete([self.config.max_utility + 1] * 3)
                if self.config.reveal_partner_utilities
                else gym.spaces.MultiDiscrete([1, 1, 1]),
                "last_partner_act": gym.spaces.Discrete(5),
                "last_partner_offer_for_me": gym.spaces.MultiDiscrete([self.config.max_count + 1] * 3),
                "turns_remaining": gym.spaces.Discrete(self.config.max_turns + 1),
            }
        )

        # Action space: act_type and allocation for me (oA). oB is derived as counts - oA
        self.action_space = gym.spaces.Dict(
            {
                "act_type": gym.spaces.Discrete(5),
                "oA": gym.spaces.MultiDiscrete([self.config.max_count + 1] * 3),
            }
        )

        # State
        self.counts: np.ndarray = np.zeros(3, dtype=int)
        self.uA: np.ndarray = np.zeros(3, dtype=int)
        self.uB: np.ndarray = np.zeros(3, dtype=int)
        self.turns_remaining: int = self.config.max_turns
        self.is_agent_turn: bool = True
        self.last_partner_act: int = self.ACT_DISAGREE
        self.last_partner_offer_for_me: np.ndarray = np.zeros(3, dtype=int)
        self.done: bool = False
        self.last_agent_offer_for_me: np.ndarray = np.zeros(3, dtype=int)
        self.last_agent_offer_for_partner: np.ndarray = np.zeros(3, dtype=int)

    # Dataset loading
    def _maybe_load_dataset(self) -> None:
        if not self.config.use_dataset or self._datasets_loaded:
            return
        try:
            from datasets import load_dataset  # type: ignore

            script = self.config.dataset_script_path
            if script is None:
                # default to local path examples/../deal_or_no_dialog_main/deal_or_no_dialog.py
                script = "deal_or_no_deal/deal_or_no_dialog_main/deal_or_no_dialog.py"
            self._dataset = load_dataset(script, name=self.config.dataset_config_name)
            self._datasets_loaded = True
        except Exception:
            self._dataset = None
            self._datasets_loaded = True

    def seed(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):  # type: ignore[override]
        if seed is not None:
            self.seed(seed)
        self._maybe_load_dataset()

        # Sample context
        self.counts, self.uA, self.uB = self._sample_context()
        self.turns_remaining = self.config.max_turns

        # Randomize who starts
        if self.config.agent_starts is None:
            self.is_agent_turn = bool(self.rng.integers(0, 2))
        else:
            self.is_agent_turn = bool(self.config.agent_starts)

        self.last_partner_act = self.ACT_DISAGREE
        self.last_partner_offer_for_me = np.zeros(3, dtype=int)
        self.done = False

        # If partner starts, simulate their move until it's our turn
        if not self.is_agent_turn and not self.done:
            self._partner_move()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: Dict[str, Any]):  # type: ignore[override]
        if self.done:
            raise RuntimeError("Episode already done. Call reset().")
        assert self._validate_action(action), "Invalid action"

        reward = 0.0
        terminated = False
        self.turns_remaining = max(0, self.turns_remaining - 1)

        act_type = int(action["act_type"])
        oA = np.array(action.get("oA", [0, 0, 0]), dtype=int)
        oA = np.clip(oA, 0, self.counts)

        # Apply agent action
        # Derive oB to respect item conservation
        oB = self.counts - oA

        if act_type in (self.ACT_PROPOSE, self.ACT_INSIST):
            # Store as last offer for partner (from us)
            self.last_agent_offer_for_partner = oB.copy()
            self.last_agent_offer_for_me = oA.copy()
            # Partner response
            self.is_agent_turn = False
            self._partner_move(agent_offer_for_partner=oB, agent_offer_for_me=oA, insisted=(act_type == self.ACT_INSIST))
        elif act_type == self.ACT_AGREE:
            # Can only agree if partner made a concrete offer
            if self.last_partner_act in (self.ACT_PROPOSE, self.ACT_INSIST):
                agreed_oA = self.last_partner_offer_for_me.copy()
                reward = float(np.dot(self.uA, agreed_oA))
                terminated = True
            else:
                # Invalid agree with no offer
                reward = -0.01
        elif act_type == self.ACT_DISAGREE:
            # No state change; partner responds
            self.is_agent_turn = False
            self._partner_move()
        elif act_type == self.ACT_END:
            # Final selection phase: agent may provide final_oA in action; partner chooses final_oB
            final_oA = np.array(action.get("final_oA", self.last_agent_offer_for_me), dtype=int)
            final_oA = np.clip(final_oA, 0, self.counts)
            # Partner's final choice: use last counter-offer if available, else greedy
            if self.last_partner_act in (self.ACT_PROPOSE, self.ACT_INSIST):
                # Partner previously proposed oA_for_me; their implied oB is counts - oA_for_me
                final_oB = self.counts - self.last_partner_offer_for_me
            else:
                # Greedy by uB
                idx_order = list(np.argsort(-self.uB))
                final_oB = np.zeros(3, dtype=int)
                rem = self.counts.copy()
                for idx in idx_order:
                    take = int(rem[idx])
                    final_oB[idx] = take
                    rem[idx] -= take
            if np.all(final_oA + final_oB == self.counts):
                reward = float(np.dot(self.uA, final_oA))
            terminated = True

        if self.turns_remaining == 0 and not terminated:
            terminated = True

        self.done = terminated

        obs = self._get_observation()
        info = self._get_info()
        return self._return_step(obs, reward, self.done, info)

    # Partner policy (simple heuristic)
    def _partner_move(self, agent_offer_for_partner: Optional[np.ndarray] = None, agent_offer_for_me: Optional[np.ndarray] = None, insisted: bool = False) -> None:
        if self.done or self.turns_remaining <= 0:
            return
        # Simple acceptance rule: accept if agent's offer gives partner utility >= threshold
        threshold = 0.5 if insisted else 0.55
        if agent_offer_for_partner is not None:
            partner_util = float(np.dot(self.uB, agent_offer_for_partner))
            max_partner_util = float(np.dot(self.uB, self.counts))
            if max_partner_util > 0 and partner_util / max_partner_util >= threshold:
                # accept (which for us means episode ends with agent's proposed split)
                self.last_partner_act = self.ACT_AGREE
                self.last_partner_offer_for_me = agent_offer_for_me.copy() if agent_offer_for_me is not None else np.zeros(3, dtype=int)
                # Agent gets its proposed share
                self.turns_remaining = max(0, self.turns_remaining - 1)
                self.done = True
                return

        # Otherwise, counter-propose: maximize partner utility subject to counts, but keep some for agent
        # Greedy by utility weight
        idx_order = list(np.argsort(-self.uB))
        oB = np.zeros(3, dtype=int)
        remaining = self.counts.copy()
        for idx in idx_order:
            # leave at least 1 item overall if possible
            take = int(max(0, remaining[idx] - 1))
            oB[idx] = take
            remaining[idx] -= take
        # ensure agent gets at least 0 for all; oA = counts - oB
        oA = self.counts - oB
        self.last_partner_act = self.ACT_PROPOSE
        self.last_partner_offer_for_me = oA
        self.turns_remaining = max(0, self.turns_remaining - 1)
        self.is_agent_turn = True

    # Sampling contexts
    def _sample_context(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.config.use_dataset and self._dataset is not None:
            split = "train" if "train" in self._dataset else list(self._dataset.keys())[0]
            idx = int(self.rng.integers(0, len(self._dataset[split])))
            row = self._dataset[split][idx]
            if self.config.dataset_config_name == "dialogues":
                counts = np.array(row["input"]["count"], dtype=int)
                uA = np.array(row["input"]["value"], dtype=int)
                uB = np.array(row["partner_input"]["value"], dtype=int)
            else:  # self_play has only input
                counts = np.array(row["input"]["count"], dtype=int)
                # Sample utilities uniformly
                uA = self.rng.integers(self.config.min_utility, self.config.max_utility + 1, size=3)
                uB = self.rng.integers(self.config.min_utility, self.config.max_utility + 1, size=3)
            counts = np.clip(counts, self.config.min_count, self.config.max_count)
            uA = np.clip(uA, self.config.min_utility, self.config.max_utility)
            uB = np.clip(uB, self.config.min_utility, self.config.max_utility)
            if self.config.normalize_utilities_to_max_points and self.config.max_points_budget > 0:
                uA = self._normalize_to_budget(counts, uA, self.config.max_points_budget)
                uB = self._normalize_to_budget(counts, uB, self.config.max_points_budget)
            return counts, uA, uB
        # Uniform sampling fallback
        counts = self.rng.integers(self.config.min_count, self.config.max_count + 1, size=3)
        counts = np.maximum(counts, 0)
        uA = self.rng.integers(self.config.min_utility, self.config.max_utility + 1, size=3)
        uB = self.rng.integers(self.config.min_utility, self.config.max_utility + 1, size=3)
        if self.config.normalize_utilities_to_max_points and self.config.max_points_budget > 0:
            uA = self._normalize_to_budget(counts, uA, self.config.max_points_budget)
            uB = self._normalize_to_budget(counts, uB, self.config.max_points_budget)
        return counts, uA, uB

    def _normalize_to_budget(self, counts: np.ndarray, utilities: np.ndarray, budget: int) -> np.ndarray:
        max_gain = int(np.dot(utilities, counts))
        if max_gain <= budget or max_gain <= 0:
            return utilities
        # scale down utilities
        scale = max(1, int(np.ceil(max_gain / float(budget))))
        scaled = np.floor_divide(utilities, scale)
        # Ensure at least one non-zero if original had
        if max_gain > 0 and scaled.sum() == 0:
            idx = int(np.argmax(utilities))
            scaled[idx] = 1
        return scaled.astype(int)

    # Utilities
    def _validate_action(self, action: Dict[str, Any]) -> bool:
        if not isinstance(action, dict):
            return False
        if "act_type" not in action or not isinstance(action["act_type"], (int, np.integer)):
            return False
        act_type = int(action["act_type"])
        if act_type < 0 or act_type > 4:
            return False
        if act_type in (self.ACT_PROPOSE, self.ACT_INSIST):
            if "oA" not in action:
                return False
        return True

    def _get_observation(self) -> Dict[str, Any]:
        partner_utils_obs = self.uB if self.config.reveal_partner_utilities else np.array([0, 0, 0], dtype=int)
        return {
            "counts": self.counts.astype(int),
            "my_utilities": self.uA.astype(int),
            "partner_utilities": partner_utils_obs.astype(int),
            "last_partner_act": int(self.last_partner_act),
            "last_partner_offer_for_me": self.last_partner_offer_for_me.astype(int),
            "turns_remaining": int(self.turns_remaining),
        }

    def _get_info(self) -> Dict[str, Any]:
        # act_type mask: agree valid only if last partner proposed or insisted
        mask = np.zeros(5, dtype=np.int8)
        mask[[self.ACT_PROPOSE, self.ACT_INSIST, self.ACT_DISAGREE, self.ACT_END]] = 1
        if self.last_partner_act in (self.ACT_PROPOSE, self.ACT_INSIST):
            mask[self.ACT_AGREE] = 1
        return {
            "action_mask": mask,
            "oA_max": self.counts.astype(int),
        }

    def render(self, mode: str = "human"):
        text = (
            f"counts={self.counts.tolist()}, uA={self.uA.tolist()}, turns_remaining={self.turns_remaining}, "
            f"last_partner_act={self.last_partner_act}, last_partner_offer_for_me={self.last_partner_offer_for_me.tolist()}"
        )
        if mode == "ansi":
            return text
        print(text)

    def _return_step(self, obs, reward, done, info):
        try:
            return obs, reward, done, False, info
        except Exception:
            return obs, reward, done, info


