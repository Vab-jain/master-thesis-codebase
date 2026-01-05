"""PPO training with Expert/Random hints.

Implements a minimal, single-env PPO loop with:
- Expert model hints from expert_seed123_turns10/ or random hints
- Actor-critic model with action and oA heads
- GAE(λ), PPO-clip, value loss, entropy bonus
- END action masking when `turns_remaining > disable_end_above_turns`
- Logging to CSV + plots, and fail-fast on excessive hint failures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from deal_or_no_deal_env.env import NegotiationEnv, NegotiationConfig
from env_wrappers.hint_injector import HintInjectorEnv
from models.policy_with_hints import PolicyWithHints


@dataclass
class PPOHParams:
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 2
    minibatch_size: int = 64
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 128
    updates: int = 2


@dataclass
class FailurePolicy:
    stop_on_ratio_gt: float = 0.5
    min_total_calls: int = 50


@dataclass
class HintsConfig:
    k: int = 1
    p: float = 0.5
    prompt_path: str = "configs/llm_prompt.txt"
    provider: str = "expert"  # "expert" or "random"
    expert_ckpt: str = "runs/expert_seed123_turns10/expert_best.pt"
    failure_policy: FailurePolicy = field(default_factory=FailurePolicy)


@dataclass
class TrainConfig:
    seed: int = 42
    max_turns: int = 10
    disable_end_above_turns: int = 2
    rolling_window: int = 100
    save_every_episodes: int = 100
    ppo: PPOHParams = field(default_factory=PPOHParams)
    hints: HintsConfig = field(default_factory=HintsConfig)
    logging_dir: str = "runs/ppo_expert_scaffold"


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dict_to_train_config(cfg: Dict[str, Any]) -> TrainConfig:
    env_cfg = cfg.get("env", {}) or {}
    hints_cfg = cfg.get("hints", {}) or {}
    ppo_cfg = (cfg.get("training", {}) or {}).get("ppo", {}) or {}
    training_cfg = cfg.get("training", {}) or {}
    logging_cfg = cfg.get("logging", {}) or {}

    failure = FailurePolicy(
        stop_on_ratio_gt=float(((hints_cfg.get("failure_policy", {}) or {}).get("stop_on_ratio_gt", 0.5))),
        min_total_calls=int(((hints_cfg.get("failure_policy", {}) or {}).get("min_total_calls", 50))),
    )
    hints = HintsConfig(
        k=int(hints_cfg.get("k", 1)),
        p=float(hints_cfg.get("p", 0.5)),
        prompt_path=str(hints_cfg.get("prompt_path", "configs/llm_prompt.txt")),
        provider=str(hints_cfg.get("provider", "expert")),
        expert_ckpt=str(hints_cfg.get("expert_ckpt", "runs/expert_seed123_turns10/expert_best.pt")),
        failure_policy=failure,
    )
    ppo = PPOHParams(
        learning_rate=float(ppo_cfg.get("learning_rate", 3.0e-4)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_coef=float(ppo_cfg.get("clip_coef", 0.2)),
        update_epochs=int(ppo_cfg.get("update_epochs", 2)),
        minibatch_size=int(ppo_cfg.get("minibatch_size", 64)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.0)),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        rollout_steps=int(ppo_cfg.get("rollout_steps", 128)),
        updates=int(ppo_cfg.get("updates", 2)),
    )
    return TrainConfig(
        seed=int(cfg.get("seed", 42)),
        max_turns=int(env_cfg.get("max_turns", 10)),
        disable_end_above_turns=int(training_cfg.get("disable_end_above_turns", 2)),
        rolling_window=int(training_cfg.get("rolling_window", 100)),
        save_every_episodes=int(training_cfg.get("save_every_episodes", 100)),
        ppo=ppo,
        hints=hints,
        logging_dir=str(logging_cfg.get("output_dir", "runs/ppo_expert_scaffold")),
    )


def make_env(cfg: TrainConfig, use_hints: bool = True) -> HintInjectorEnv | NegotiationEnv:
    env = NegotiationEnv(
        NegotiationConfig(
            max_turns=cfg.max_turns,
            use_dataset=False,
        )
    )
    
    if use_hints:
        # Use built-in expert or random provider - no external client needed
        wrapped = HintInjectorEnv(
            env, 
            client=None,  # No external client needed for expert/random
            k=cfg.hints.k, 
            prompt_path=cfg.hints.prompt_path, 
            provider=cfg.hints.provider,
            expert_ckpt=cfg.hints.expert_ckpt
        )
        return wrapped
    return env


def obs_to_input(obs: Dict[str, Any], hint: Dict[str, Any]) -> np.ndarray:
    counts = np.array(obs["counts"], dtype=np.float32)
    my_utils = np.array(obs["my_utilities"], dtype=np.float32)
    partner_utils = np.array(obs["partner_utilities"], dtype=np.float32)
    last_act = int(obs["last_partner_act"]) if isinstance(obs["last_partner_act"], (int, np.integer)) else int(obs["last_partner_act"])  # type: ignore[arg-type]
    last_act_oh = np.zeros(5, dtype=np.float32)
    last_act_oh[min(4, max(0, last_act))] = 1.0
    last_offer = np.array(obs["last_partner_offer_for_me"], dtype=np.float32)
    tr = float(obs["turns_remaining"]) if obs.get("turns_remaining") is not None else 0.0
    base = np.concatenate([counts, my_utils, partner_utils, last_act_oh, last_offer, np.array([tr], dtype=np.float32)])
    # hint feats
    act_onehot = np.array(hint.get("act_onehot", [0.0] * 5), dtype=np.float32)
    oA_norm = np.array(hint.get("oA_norm", [0.0] * 3), dtype=np.float32)
    confidence = np.array([hint.get("confidence", 0.0)], dtype=np.float32)
    h_avail = np.array([float(hint.get("h_avail", 0))], dtype=np.float32)
    return np.concatenate([base, act_onehot, oA_norm, confidence, h_avail])


def compute_gae(rewards: List[float], dones: List[bool], values: List[float], gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - float(dones[t]) if t == T - 1 else 1.0 - float(dones[t + 1])
        nextvalue = values[t] if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns


def _select_best_hints_run() -> Dict[str, Any]:
    # Scan for runs/hints_eval*/summary.txt and pick the best by act_accuracy, then lowest mae
    import glob

    candidates = []
    for path in glob.glob("runs/hints_eval*/summary.txt"):
        try:
            text = Path(path).read_text(encoding="utf-8")
            lines = {k.split("=")[0]: k.split("=", 1)[1] for k in text.strip().splitlines() if "=" in k}
            acc = float(lines.get("act_accuracy", "0"))
            mae_str = lines.get("oA_mae_per_item", "[9,9,9]")
            mae_vals = [float(x) for x in mae_str.strip("[]").split(",")]
            candidates.append((acc, sum(mae_vals), path))
        except Exception:
            continue
    if not candidates:
        return {"source": "none"}
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return {"source": candidates[0][2], "act_accuracy": candidates[0][0]}


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO + Expert/Random-Hints Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config",
    )
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--no_hints", action="store_true")
    parser.add_argument("--total_episodes", type=int, default=250)
    parser.add_argument("--total_env_steps", type=int, default=0, help="Stop after this many env steps if >0")
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--provider", type=str, default=None, choices=["expert", "random"])
    parser.add_argument("--expert_ckpt", type=str, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--save_every_episodes", type=int, default=None)
    args = parser.parse_args()

    cfg_dict = load_yaml_config(args.config)
    cfg = dict_to_train_config(cfg_dict)
    # CLI overrides
    if args.provider is not None:
        cfg.hints.provider = str(args.provider)
    if args.expert_ckpt is not None:
        cfg.hints.expert_ckpt = str(args.expert_ckpt)
    if args.k is not None:
        cfg.hints.k = int(args.k)

    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.logging_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    returns_csv = out_dir / "train_returns.csv"
    losses_csv = out_dir / "losses.csv"
    llm_csv = out_dir / "llm_calls.csv"

    # selection.json and snapshots
    sel = _select_best_hints_run()
    sel["prompt_path"] = cfg_dict.get("hints", {}).get("prompt_path", "configs/llm_prompt.txt")
    sel["hint_provider"] = cfg.hints.provider
    sel["expert_ckpt"] = cfg.hints.expert_ckpt
    (out_dir / "selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    # snapshot config
    (out_dir / "config_snapshot.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    # snapshot prompt
    try:
        prompt_path = Path(cfg_dict.get("hints", {}).get("prompt_path", "configs/llm_prompt.txt"))
        (out_dir / "prompt_snapshot.txt").write_text(prompt_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    env = make_env(cfg, use_hints=(not args.no_hints))
    run_seed = int(args.seed) if args.seed is not None else int(cfg.seed)
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)

    input_dim = 3 + 3 + 3 + 5 + 3 + 1 + 5 + 3 + 1 + 1  # base + hint
    policy = PolicyWithHints(input_dim=input_dim, hidden=64)
    cfg.ppo.ent_coef = float(args.entropy_coef)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.learning_rate)

    # Storage
    obs_buf: List[np.ndarray] = []
    hint_buf: List[Dict[str, Any]] = []
    info_buf: List[Dict[str, Any]] = []
    actions_buf: List[Dict[str, Any]] = []
    logprobs_buf: List[float] = []
    values_buf: List[float] = []
    rewards_buf: List[float] = []
    dones_buf: List[bool] = []

    episode_return = 0.0
    returns_history: List[float] = []
    # Episode CSV logging setup
    from collections import deque
    rolling_win = max(1, cfg.rolling_window)
    rw = deque(maxlen=rolling_win)
    if not returns_csv.exists():
        with returns_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "return", f"rolling_mean_return_{cfg.rolling_window}", "step"])

    obs, info = env.reset(seed=cfg.seed)
    episodes_target = max(1, int(args.total_episodes))
    steps_target = int(args.total_env_steps) if int(args.total_env_steps) > 0 else None
    update = 0
    acts_episode: List[int] = []
    turns_used: List[int] = []
    agree_flags: List[bool] = []
    
    env_steps = 0
    while (len(returns_history) < episodes_target) and (steps_target is None or env_steps < steps_target):
        # Rollout
        hint_latency_sum = 0.0
        hint_latency_count = 0
        for step in range(cfg.ppo.rollout_steps):
            hint = info.get("hint_features", {})

            x_np = obs_to_input(obs, hint)
            x = torch.from_numpy(x_np).unsqueeze(0)
            out = policy(x)

            # Masks
            act_mask = np.array(info.get("action_mask", [1, 1, 1, 1, 1]), dtype=np.float32)
            # Disable END if turns_remaining too high
            if obs.get("turns_remaining", 0) > cfg.disable_end_above_turns:
                act_mask[4] = 0.0
            act_mask_t = torch.from_numpy(act_mask).unsqueeze(0)
            action_logits = PolicyWithHints.apply_masks(out["action_logits"], act_mask_t)
            dist_act = Categorical(logits=action_logits)
            act_idx = int(dist_act.sample().item())
            logp_act = float(dist_act.log_prob(torch.tensor([act_idx])).item())
            ent_act = float(dist_act.entropy().mean().item())

            # oA heads with caps
            oA_max = np.array(info.get("oA_max", [1, 1, 1]), dtype=np.int64)
            oA_vals: List[int] = []
            logp_oa = 0.0
            ent_oa = 0.0
            for i in range(3):
                logits_i = PolicyWithHints.apply_oa_caps(out[f"oA_logits_{i}"], int(oA_max[i]))
                dist_i = Categorical(logits=logits_i)
                idx_i = int(dist_i.sample().item())
                oA_vals.append(idx_i)
                if act_idx in (0, 1):  # only count when propose/insist
                    logp_oa += float(dist_i.log_prob(torch.tensor([idx_i])).item())
                    ent_oa += float(dist_i.entropy().mean().item())

            value = float(out["value"].item())

            action_dict = {"act_type": act_idx}
            if act_idx in (0, 1):
                action_dict["oA"] = oA_vals

            next_obs, reward, terminated, truncated, next_info = env.step(action_dict)
            env_steps += 1

            obs_buf.append(x_np)
            hint_buf.append(hint)
            info_buf.append(info)
            actions_buf.append(action_dict)
            logprobs_buf.append(logp_act + logp_oa)
            values_buf.append(value)
            rewards_buf.append(float(reward))
            done = bool(terminated or truncated)
            dones_buf.append(done)
            # Track hint latency
            if info.get("hint_called"):
                hint_latency_sum += float(info.get("hint_latency_ms", 0.0))
                hint_latency_count += 1

            episode_return += float(reward)
            if done:
                returns_history.append(episode_return)
                rw.append(episode_return)
                with returns_csv.open("a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([len(returns_history), float(episode_return), float(sum(rw) / len(rw)), int(update)])
                episode_return = 0.0
                next_obs, next_info = env.reset()
                # per-episode summaries
                # Determine if agreed or ended
                agreed = bool(env.last_partner_act == env.ACT_AGREE)
                agree_flags.append(agreed)
                # end-rate based on last taken action if available
                try:
                    last_action = actions_buf[-1] if len(actions_buf) > 0 else {"act_type": 3}
                    acts_episode.append(int(last_action.get("act_type", 3)))
                except Exception:
                    acts_episode.append(3)
                # turns used approximation: max_turns - turns_remaining
                try:
                    tu = int(cfg.max_turns - (next_info.get("turns_remaining", cfg.max_turns)))
                except Exception:
                    tu = cfg.max_turns
                turns_used.append(max(1, tu))
            if (len(returns_history) >= episodes_target) or (steps_target is not None and env_steps >= steps_target):
                break
        if (len(returns_history) >= episodes_target) or (steps_target is not None and env_steps >= steps_target):
            break

        obs, info = next_obs, next_info

        # GAE + returns
        adv, rets = compute_gae(rewards_buf, dones_buf, values_buf, cfg.ppo.gamma, cfg.ppo.gae_lambda)
        adv_t = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8))
        rets_t = torch.from_numpy(rets)
        obs_t = torch.from_numpy(np.stack(obs_buf, axis=0))

        # Recompute policy outputs for PPO and apply minibatching
        bsz_total = obs_t.shape[0]
        idxs = np.arange(bsz_total)
        minibatch_size = max(1, min(int(cfg.ppo.minibatch_size), bsz_total))
        update_epochs = max(1, int(cfg.ppo.update_epochs))

        out_all = policy(obs_t)
        values_all = out_all["value"].detach()

        def compute_logprob_and_entropy(indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
            lp_new = torch.zeros(len(indices))
            entropies_mb = torch.zeros(len(indices))
            for j, t in enumerate(indices):
                info_t = info_buf[int(t)]
                obsd = actions_buf[int(t)]
                act_taken = int(obsd["act_type"]) if "act_type" in obsd else int(obsd.get("act", 0))
                mask_np = np.array(info_t.get("action_mask", [1, 1, 1, 1, 1]), dtype=np.float32)
                if info_t.get("turns_remaining", None) is not None:
                    if info_t["turns_remaining"] > cfg.disable_end_above_turns:
                        mask_np[4] = 0.0
                act_logits_t = PolicyWithHints.apply_masks(out_all["action_logits"][int(t)], torch.from_numpy(mask_np))
                dist_t = Categorical(logits=act_logits_t)
                lp_new[j] = dist_t.log_prob(torch.tensor(act_taken))
                ent = dist_t.entropy()
                if act_taken in (0, 1) and "oA" in obsd:
                    oA_max = np.array(info_t.get("oA_max", [1, 1, 1]), dtype=np.int64)
                    for i in range(3):
                        logits_i = PolicyWithHints.apply_oa_caps(out_all[f"oA_logits_{i}"][int(t)], int(oA_max[i]))
                        dist_i = Categorical(logits=logits_i)
                        lp_new[j] = lp_new[j] + dist_i.log_prob(torch.tensor(int(obsd["oA"][i])))
                        ent = ent + dist_i.entropy()
                entropies_mb[j] = ent
            return lp_new, entropies_mb

        old_logprobs_t = torch.tensor(logprobs_buf)

        for _ep in range(update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, bsz_total, minibatch_size):
                mb_idx = idxs[start : min(start + minibatch_size, bsz_total)]
                lp_new_mb, ent_mb = compute_logprob_and_entropy(mb_idx)
                ratio = (lp_new_mb - old_logprobs_t[mb_idx]).exp()
                adv_mb = adv_t[mb_idx]
                pg1 = ratio * adv_mb
                pg2 = torch.clamp(ratio, 1.0 - cfg.ppo.clip_coef, 1.0 + cfg.ppo.clip_coef) * adv_mb
                policy_loss = -torch.min(pg1, pg2).mean()

                value_loss = 0.5 * (rets_t[mb_idx] - values_all[mb_idx]).pow(2).mean()
                entropy_loss = -ent_mb.mean()
                approx_kl = (old_logprobs_t[mb_idx] - lp_new_mb).mean().item()
                clip_fraction = (torch.gt(torch.abs(ratio - 1.0), cfg.ppo.clip_coef)).float().mean().item()

                loss = policy_loss + cfg.ppo.vf_coef * value_loss + cfg.ppo.ent_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.ppo.max_grad_norm)
                optimizer.step()

        # Logging
        rolling = float(np.mean(returns_history[-cfg.rolling_window:])) if returns_history else 0.0
        # losses only; returns per-episode are written above
        with losses_csv.open("a", newline="") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["step", "policy_loss", "value_loss", "entropy"])
            w.writerow([update, float(policy_loss.item()), float(value_loss.item()), float(-entropy_loss.item())])
        with llm_csv.open("a", newline="") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["step", "total_calls", "failed_calls", "ratio", "avg_latency_ms", "hint_source"])
            total = int(getattr(env, "total_hint_calls", 0))
            failed = int(getattr(env, "failed_hint_calls", 0))
            retries = int(getattr(env, "total_retries", 0))
            illegal = int(getattr(env, "illegal_suggestions", 0))
            successes = int(getattr(env, "hint_available_successes", 0))
            pct = float((successes / total) if total > 0 else 0.0)
            ratio = float(failed / total) if total > 0 else 0.0
            avg_lat = float(hint_latency_sum / hint_latency_count) if hint_latency_count > 0 else 0.0
            src = cfg.hints.provider
            w.writerow([update, total, failed, ratio, avg_lat, src])
        try:
            print(f"[LLM] update={update} total_calls={total} retries={retries} failed={failed} illegal={illegal} hint_available_pct={pct:.2%} avg_latency_ms={avg_lat:.1f}")
        except Exception:
            pass

        # periodic checkpoint by episodes trained
        save_every = int(args.save_every_episodes) if args.save_every_episodes is not None else int(cfg.save_every_episodes)
        if save_every > 0 and len(returns_history) > 0 and (len(returns_history) % save_every == 0):
            torch.save({"policy": policy.state_dict()}, ckpt_dir / f"ckpt_ep_{len(returns_history)}.pt")

        # Fail-fast
        fp = cfg.hints.failure_policy
        total = int(getattr(env, "total_hint_calls", 0))
        failed = int(getattr(env, "failed_hint_calls", 0))
        ratio = float(failed / total) if total > 0 else 0.0
        if total >= fp.min_total_calls and ratio > fp.stop_on_ratio_gt:
            # Save checkpoint and status and exit
            torch.save({"policy": policy.state_dict()}, ckpt_dir / "policy.pt")
            with (out_dir / "status.json").open("w", encoding="utf-8") as f:
                json.dump({"status": "HINT_FAILURE", "total_hint_calls": total, "failed_hint_calls": failed, "ratio": ratio}, f)
            return

        # Clear buffers for next update
        obs_buf.clear()
        hint_buf.clear()
        info_buf.clear()
        actions_buf.clear()
        logprobs_buf.clear()
        values_buf.clear()
        rewards_buf.clear()
        dones_buf.clear()
        update += 1

    # Save final checkpoint and status
    torch.save({"policy": policy.state_dict()}, ckpt_dir / "ckpt_final.pt")
    (out_dir / "status.json").write_text(json.dumps({"status": "OK"}, indent=2), encoding="utf-8")

    # Final statistics
    from evaluation.metrics import episode_metrics  # type: ignore
    mets = episode_metrics(returns_history, acts_episode, turns_used, agree_flags)
    final = {
        **mets,
        "episodes_trained": int(len(returns_history)),
        "seed": int(run_seed),
        "algo": "ppo",
        "hint_mode": cfg.hints.provider,
        "k": int(cfg.hints.k),
        "hint_provider": cfg.hints.provider,
        "expert_ckpt": cfg.hints.expert_ckpt if cfg.hints.provider == "expert" else None,
        "failure_ratio": float((getattr(env, "failed_hint_calls", 0) / getattr(env, "total_hint_calls", 1)) if getattr(env, "total_hint_calls", 0) > 0 else 0.0),
    }
    with (out_dir / "final_stats.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    # CSV
    with (out_dir / "final_stats.csv").open("w", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=list(final.keys()))
        w.writeheader(); w.writerow(final)

    # Save plots at the end
    try:
        import matplotlib.pyplot as plt  # type: ignore

        # Rolling return plot
        plt.figure(figsize=(6, 3))
        rs = [np.mean(returns_history[max(0, i - cfg.rolling_window): i + 1]) for i in range(len(returns_history))]
        plt.plot(rs)
        plt.title("Rolling Return")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.tight_layout()
        plt.savefig(out_dir / "rolling_return.png")
        plt.close()

        # Losses plot
        # Read losses.csv
        xs: List[int] = []
        pl: List[float] = []
        vl: List[float] = []
        ent: List[float] = []
        with (losses_csv).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step_key = "update" if "update" in row else ("step" if "step" in row else None)
                if step_key is None:
                    continue
                xs.append(int(row[step_key]))
                pl.append(float(row["policy_loss"]))
                vl.append(float(row["value_loss"]))
                ent.append(float(row["entropy"]))
        plt.figure(figsize=(6, 3))
        plt.plot(xs, pl, label="policy")
        plt.plot(xs, vl, label="value")
        plt.plot(xs, ent, label="entropy")
        plt.legend()
        plt.title("Losses")
        plt.xlabel("Update")
        plt.tight_layout()
        plt.savefig(out_dir / "losses.png")
        plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
