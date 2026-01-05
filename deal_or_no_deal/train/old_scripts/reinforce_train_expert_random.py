#!/usr/bin/env python3
"""REINFORCE training with Expert/Random hints.

Implements REINFORCE policy gradient training with:
- Expert model hints from expert_seed123_turns10/ or random hints
- Simple policy network with action and oA heads
- Vanilla policy gradient with return-based rewards
- Logging to CSV + plots
"""

from __future__ import annotations

import os
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import csv
import matplotlib.pyplot as plt

from deal_or_no_deal_env import register_deal_or_no_deal
from deal_or_no_deal_env.env import NegotiationConfig
from env_wrappers.hint_injector import HintInjectorEnv


def make_env(max_turns: int, provider: str, expert_ckpt: str, k: int, prompt_path: str):
    env_id = register_deal_or_no_deal()
    base = gym.make(env_id, config=NegotiationConfig(max_turns=max_turns, use_dataset=False))
    return HintInjectorEnv(
        base, 
        client=None,  # No external client needed for expert/random
        k=k, 
        prompt_path=prompt_path, 
        provider=provider,
        expert_ckpt=expert_ckpt
    )


class PolicyNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        # base obs flat + hint features(10)
        input_dim = 3 + 3 + 1 + 3 + 1 + 10
        self.backbone = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.act_head = nn.Linear(hidden, 5)
        self.o_heads = nn.ModuleList([nn.Linear(hidden, 5) for _ in range(3)])

    def forward(self, x):
        h = self.backbone(x)
        return self.act_head(h), [h2(h) for h2 in self.o_heads]

    def obs_to_input(self, obs: Dict[str, Any], hint: Dict[str, Any]) -> torch.Tensor:
        counts = np.array(obs["counts"], dtype=np.float32)
        my_utils = np.array(obs["my_utilities"], dtype=np.float32)
        last_act = np.array([obs["last_partner_act"]], dtype=np.float32)
        last_offer = np.array(obs["last_partner_offer_for_me"], dtype=np.float32)
        turns = np.array([obs["turns_remaining"]], dtype=np.float32)
        # hint feats
        act_onehot = np.array(hint.get("act_onehot", [0]*5), dtype=np.float32)
        oA_norm = np.array(hint.get("oA_norm", [0]*3), dtype=np.float32)
        conf = np.array([hint.get("confidence", 0.0)], dtype=np.float32)
        hav = np.array([float(hint.get("h_avail", 0))], dtype=np.float32)
        x = np.concatenate([counts, my_utils, last_act, last_offer, turns, act_onehot, oA_norm, conf, hav]).astype(np.float32)
        return torch.from_numpy(x)

    def sample(self, obs: Dict[str, Any], info: Dict[str, Any]) -> tuple[Dict[str, Any], torch.Tensor]:
        hint = info.get("hint_features", {})
        x = self.obs_to_input(obs, hint)
        act_logits, o_logits = self.forward(x)
        act_mask = torch.tensor(info.get("action_mask", [1,1,1,1,1]), dtype=torch.float32)
        act_probs = torch.softmax(act_logits, dim=-1) * act_mask
        act_probs = act_probs / (act_probs.sum() + 1e-8)
        act_dist = torch.distributions.Categorical(probs=act_probs)
        act = int(act_dist.sample().item())
        logp = act_dist.log_prob(torch.tensor(act))
        oA = [0,0,0]
        if act in (0,1):
            for i, logits in enumerate(o_logits):
                probs = torch.softmax(logits, dim=-1)
                cap = int(info.get("oA_max", [4,4,4])[i])
                mask = torch.zeros_like(probs); mask[:cap+1] = 1.0
                probs = probs * mask; probs = probs / (probs.sum() + 1e-8)
                dist = torch.distributions.Categorical(probs=probs)
                val = int(dist.sample().item())
                oA[i] = val
                logp = logp + dist.log_prob(torch.tensor(val))
        return {"act_type": act, "oA": oA}, logp


def train(out_dir: str, total_episodes: int = 2000, seed: int = 1234, max_turns: int = 10,
          provider: str = "expert", expert_ckpt: str = "runs/expert_seed123_turns10/expert_best.pt",
          k: int = 1, prompt_path: str = "configs/llm_prompt.txt", save_every_episodes: int = 100,
          total_env_steps: int = 0, entropy_coef: float = 0.01, num_envs: int = 1):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = Path(out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed); torch.manual_seed(seed)
    env = make_env(max_turns, provider, expert_ckpt, k, prompt_path)
    policy = PolicyNet()
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    returns_csv = Path(out_dir) / "train_returns.csv"
    losses_csv = Path(out_dir) / "losses.csv"
    llm_csv = Path(out_dir) / "llm_calls.csv"
    
    with returns_csv.open("w", newline="") as f:
        csv.writer(f).writerow(["episode", "return", "rolling_mean_return_100", "step"])
    with losses_csv.open("w", newline="") as f:
        csv.writer(f).writerow(["step", "policy_loss", "value_loss", "entropy"])
    with llm_csv.open("w", newline="") as f:
        csv.writer(f).writerow(["step", "total_calls", "failed_calls", "ratio", "avg_latency_ms", "hint_source"])

    rw = deque(maxlen=100)
    acts_episode = []
    agree_flags = []
    turns_used = []
    env_steps = 0
    total_env_steps = int(os.environ.get("TOTAL_ENV_STEPS_OVERRIDE", "0"))  # optional external control
    for ep in trange(total_episodes, desc=f"REINFORCE_{provider.upper()}"):
        obs, info = env.reset(seed=seed+ep)
        traj_logps = []; traj_rewards = []; steps=0; done=False
        while not done:
            # If using LLM-like providers here (expert/random are not LLM), proceed normally
            action, logp = policy.sample(obs, info)
            obs, reward, done, truncated, info = env.step(action)
            traj_logps.append(logp); traj_rewards.append(float(reward)); steps+=1
            env_steps += 1
            if truncated:
                break
            # Optional step-based termination
            if total_env_steps > 0 and env_steps >= total_env_steps:
                done = True
        returns = []
        G=0.0
        for r in reversed(traj_rewards):
            G = r + 0.99*G
            returns.append(G)
        returns = list(reversed(returns))
        returns_t = torch.tensor(returns, dtype=torch.float32)
        logps_t = torch.stack(traj_logps)
        loss = -(logps_t * (returns_t - returns_t.mean())).sum() - float(entropy_coef) * logps_t.mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        ep_ret = sum(traj_rewards)
        rw.append(ep_ret)
        
        # Log episode data
        with returns_csv.open("a", newline="") as f:
            csv.writer(f).writerow([ep+1, ep_ret, float(np.mean(rw)), ep+1])
        with losses_csv.open("a", newline="") as f:
            csv.writer(f).writerow([ep+1, float(loss.item()), float("nan"), float(0.0)])
        
        # Log hint statistics
        with llm_csv.open("a", newline="") as f:
            total = int(getattr(env, "total_hint_calls", 0))
            failed = int(getattr(env, "failed_hint_calls", 0))
            ratio = float(failed / total) if total > 0 else 0.0
            csv.writer(f).writerow([ep+1, total, failed, ratio, 0.0, provider])
        
        acts_episode.append(int(action["act_type"]))
        # Try to access the base environment through the wrapper
        try:
            base_env = env._env.env if hasattr(env._env, 'env') else env._env
            agree_flags.append(bool(base_env.last_partner_act == base_env.ACT_AGREE or action["act_type"] == base_env.ACT_AGREE))
        except AttributeError:
            # Fallback if we can't access the attributes
            agree_flags.append(False)
        turns_used.append(steps)

        if save_every_episodes > 0 and (ep + 1) % save_every_episodes == 0:
            torch.save({"policy": policy.state_dict()}, ckpt_dir / f"ckpt_ep_{ep+1}.pt")

    torch.save({"policy": policy.state_dict()}, ckpt_dir / "ckpt_final.pt")
    with (Path(out_dir)/"status.json").open("w") as f:
        json.dump({"status": "OK"}, f, indent=2)
    
    # Final stats
    from evaluation.metrics import episode_metrics  # type: ignore
    mets = episode_metrics([float(r) for r in list(rw)], acts_episode, turns_used, agree_flags)
    final = {
        **mets,
        "episodes_trained": int(total_episodes),
        "seed": int(seed),
        "algo": "reinforce",
        "hint_mode": provider,
        "k": int(k),
        "hint_provider": provider,
        "expert_ckpt": expert_ckpt if provider == "expert" else None,
        "failure_ratio": 0.0 if provider in ("random","expert") else 0.0,
    }
    with (Path(out_dir)/"final_stats.json").open("w") as f:
        json.dump(final, f, indent=2)
    with (Path(out_dir)/"final_stats.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(final.keys()))
        w.writeheader(); w.writerow(final)
    
    try:
        import pandas as pd  # type: ignore
        df = pd.read_csv(returns_csv)
        avg = float(df["return"].mean()); std = float(df["return"].std()); med = float(df["return"].median())
        with (Path(out_dir)/"eval_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["avg_return","std_return","median_return"]) ; w.writeheader(); w.writerow({"avg_return":avg,"std_return":std,"median_return":med})
    except Exception:
        pass

    # Save plots
    try:
        plt.figure(figsize=(6, 3))
        rs = [np.mean(list(rw)[max(0, i - 100): i + 1]) for i in range(len(rw))]
        plt.plot(rs)
        plt.title("Rolling Return")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "rolling_return.png")
        plt.close()
    except Exception:
        pass
    # Plot training loss curve from losses.csv
    try:
        xs_l, ys_l = [], []
        with losses_csv.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                step_val = row.get("step") or row.get("episode") or row.get("update")
                if step_val is None:
                    continue
                xs_l.append(int(step_val))
                ys_l.append(float(row.get("policy_loss", 0.0)))
        if len(xs_l) > 0:
            plt.figure(figsize=(6, 3))
            plt.plot(xs_l, ys_l, label="policy_loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(out_dir) / "training_loss.png")
            plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--total_episodes", type=int, default=2000)
    p.add_argument("--total_env_steps", type=int, default=0, help="Stop after this many env steps if >0")
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max_turns", type=int, default=10)
    p.add_argument("--provider", type=str, default="expert", choices=["expert", "random"])
    p.add_argument("--expert_ckpt", type=str, default="runs/expert_seed123_turns10/expert_best.pt")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--prompt_path", type=str, default="configs/llm_prompt.txt")
    p.add_argument("--save_every_episodes", type=int, default=100)
    args = p.parse_args()
    train(
        out_dir=args.out_dir,
        total_episodes=args.total_episodes,
        seed=args.seed,
        max_turns=args.max_turns,
        provider=args.provider,
        expert_ckpt=args.expert_ckpt,
        k=args.k,
        prompt_path=args.prompt_path,
        save_every_episodes=args.save_every_episodes,
        total_env_steps=args.total_env_steps,
        entropy_coef=args.entropy_coef,
        num_envs=args.num_envs,
    )
