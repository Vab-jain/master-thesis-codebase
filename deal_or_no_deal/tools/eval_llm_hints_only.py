#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

import numpy as np
import gymnasium as gym

from deal_or_no_deal_env import register_deal_or_no_deal
from deal_or_no_deal_env.env import NegotiationConfig
from llm.client import GROQClient
from llm.schema import parse_next_action
from llm.prompt import render_prompt_with_cap


def build_prompt_from_obs(template_path: str, obs: Dict[str, Any], info: Dict[str, Any], max_chars: int = 2000) -> str:
    def _act_token(val: int) -> str:
        mapping = {0: "PROPOSE", 1: "INSIST", 2: "AGREE", 3: "DISAGREE", 4: "END"}
        try:
            return mapping.get(int(val), "NONE")
        except Exception:
            return "NONE"

    counts = obs.get("counts", [0, 0, 0])
    my_utils = obs.get("my_utilities", [0, 0, 0])
    last_offer = obs.get("last_partner_offer_for_me", [0, 0, 0])
    last_act = obs.get("last_partner_act", 4)
    turns = obs.get("turns_remaining", 0)
    mask = info.get("action_mask", [1, 1, 1, 1, 1])
    variables = {
        "counts_csv": ",".join(str(int(x)) for x in list(counts)),
        "my_utils_csv": ",".join(str(int(x)) for x in list(my_utils)),
        "last_partner_act_token": _act_token(last_act),
        "last_offer_csv": ",".join(str(int(x)) for x in list(last_offer)),
        "turns": int(turns),
        "legal_mask_csv": ",".join(str(int(x)) for x in list(mask)),
        "p": 0.5,
        "history_str": "",
    }
    return render_prompt_with_cap(template_path, variables, max_chars)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM hints only (no training)")
    parser.add_argument("--env-id", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile")
    parser.add_argument("--prompt_path", type=str, default="configs/llm_prompt.txt")
    parser.add_argument("--show-examples", type=int, default=5)
    args = parser.parse_args()

    env_id = args.env_id or register_deal_or_no_deal()
    env = gym.make(env_id, config=NegotiationConfig(max_turns=10, use_dataset=False))
    client = GROQClient(model=args.model)

    rng = np.random.default_rng(args.seed)
    episode_rewards: List[float] = []
    failures = 0
    shown = 0
    any_valid = False

    for ep in range(1, int(args.episodes) + 1):
        obs, info = env.reset(seed=int(args.seed) + ep)
        done = False
        ret = 0.0
        steps = 0
        while not done:
            prompt = build_prompt_from_obs(args.prompt_path, obs, info)
            try:
                text, latency_ms = client.generate(prompt)
                parsed = parse_next_action(text)
                act = int(parsed.get("act_type", 4))
                mask = info.get("action_mask", [1, 1, 1, 1, 1])
                legal = (0 <= act < len(mask)) and (int(mask[act]) == 1)
                if not legal:
                    raise RuntimeError("illegal action from LLM")
                action = {"act_type": act}
                if act in (0, 1):
                    # Use suggested allocations if present; clamp is not allowed here; we trust LLM only for action
                    action["oA"] = [0, 0, 0]
                any_valid = True
            except Exception as exc:  # noqa: BLE001
                failures += 1
                # Print example on failure
                if shown < int(args.show_examples):
                    print(f"[EXAMPLE] obs={{counts:{obs.get('counts')}, my_utils:{obs.get('my_utilities')}, last_act:{obs.get('last_partner_act')}, last_offer:{obs.get('last_partner_offer_for_me')}, turns:{obs.get('turns_remaining')}}} mask={info.get('action_mask')}")
                    compact_prompt = " ".join(prompt.split())[:200]
                    print(f"[PROMPT] {compact_prompt}")
                    print(f"[LLM] error={exc}")
                    shown += 1
                # End episode early on failure per constraints
                break

            # Print a few successful examples
            if shown < int(args.show_examples):
                compact_prompt = " ".join(prompt.split())[:200]
                print(f"[EXAMPLE] obs={{counts:{obs.get('counts')}, my_utils:{obs.get('my_utilities')}, last_act:{obs.get('last_partner_act')}, last_offer:{obs.get('last_partner_offer_for_me')}, turns:{obs.get('turns_remaining')}}} mask={info.get('action_mask')}")
                print(f"[PROMPT] {compact_prompt}")
                print(f"[LLM] raw={text}")
                print(f"[LLM] parsed_act={act}")
                shown += 1

            obs, reward, done, truncated, info = env.step(action)
            ret += float(reward)
            steps += 1
            if truncated:
                break

        episode_rewards.append(ret)

    avg = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    print(f"episodes={len(episode_rewards)} avg_reward={avg:.3f} rewards={episode_rewards}")
    # Summarize hint stats similar to trainers
    total_calls = int(getattr(env, "total_hint_calls", 0))
    retries = int(getattr(env, "total_retries", 0))
    failed = int(getattr(env, "failed_hint_calls", 0))
    illegal = int(getattr(env, "illegal_suggestions", 0))
    successes = int(getattr(env, "hint_available_successes", 0))
    pct = float((successes / total_calls) if total_calls > 0 else 0.0)
    print(f"llm_failures={failures} | summary: total_calls={total_calls} retries={retries} failed={failed} illegal={illegal} hint_available_pct={pct:.2%}")
    if not any_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()


