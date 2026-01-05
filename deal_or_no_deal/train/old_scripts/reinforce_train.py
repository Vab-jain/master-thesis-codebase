#!/usr/bin/env python3
import os
import json
import time
from collections import deque

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
from torch.utils.tensorboard import SummaryWriter


def make_env(cfg: NegotiationConfig):
    env_id = register_deal_or_no_deal()
    return gym.make(env_id, config=cfg)


class PolicyNet(nn.Module):
    def __init__(self, max_count=4, max_utility=10, hidden=128):
        super().__init__()
        self.max_count = max_count
        self.max_utility = max_utility
        # Input: counts(3), my_utils(3), last_partner_act(1), last_partner_offer(3), turns_remaining(1)
        input_dim = 3 + 3 + 1 + 3 + 1
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        # act logits (5)
        self.act_head = nn.Linear(hidden, 5)
        # allocation head for oA (3) via independent categorical over [0..max_count]
        self.o_heads = nn.ModuleList([nn.Linear(hidden, max_count + 1) for _ in range(3)])

    def forward(self, obs):
        # obs dict of numpy arrays
        x = np.concatenate([
            obs["counts"],
            obs["my_utilities"],
            [obs["last_partner_act"]],
            obs["last_partner_offer_for_me"],
            [obs["turns_remaining"]],
        ]).astype(np.float32)
        x = torch.from_numpy(x)
        h = self.backbone(x)
        act_logits = self.act_head(h)
        o_logits = [head(h) for head in self.o_heads]
        return act_logits, o_logits

    def sample(self, obs, oA_max, action_mask=None, disable_end_above_turns=None):
        act_logits, o_logits = self.forward(obs)
        act_probs = torch.softmax(act_logits, dim=-1)
        # Mask invalid actions
        if action_mask is not None:
            mask = torch.from_numpy(action_mask.astype(np.float32))
            act_probs = act_probs * mask
        # Heuristic: discourage 'end' early
        if disable_end_above_turns is not None and obs.get("turns_remaining", 0) > disable_end_above_turns:
            act_probs[4] = 0.0
        act_probs = act_probs / (act_probs.sum() + 1e-8)
        act_dist = torch.distributions.Categorical(probs=act_probs)
        act = int(act_dist.sample().item())
        logp_act = act_dist.log_prob(torch.tensor(act))
        # sample oA per item with cap oA_max
        oA = []
        logp_o = []
        for i, logits in enumerate(o_logits):
            probs = torch.softmax(logits, dim=-1)
            # mask values > oA_max[i]
            mask = torch.zeros_like(probs)
            mask[: int(oA_max[i]) + 1] = 1.0
            probs = probs * mask
            probs = probs / (probs.sum() + 1e-8)
            dist = torch.distributions.Categorical(probs=probs)
            val = int(dist.sample().item())
            oA.append(val)
            logp_o.append(dist.log_prob(torch.tensor(val)))
        logp = logp_act + torch.stack(logp_o).sum()
        return {"act_type": act, "oA": oA}, logp


def train(
    total_episodes=2000,
    gamma=0.99,
    lr=3e-4,
    entropy_coef=0.01,
    max_turns=10,
    use_dataset=True,
    dataset_config_name="dialogues",
    dataset_script_path="./deal_or_no_dialog/deal_or_no_dialog.py",
    out_dir="runs/reinforce_seed412",
    seed=1234,
    rolling_window=100,
    save_every_episodes: int = 100,
    total_env_steps: int = 0,
    num_envs: int = 1,
):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tb = SummaryWriter(log_dir=out_dir)

    cfg = NegotiationConfig(
        use_dataset=use_dataset,
        dataset_script_path=dataset_script_path,
        dataset_config_name=dataset_config_name,
        max_turns=max_turns,
        normalize_utilities_to_max_points=True,
        max_points_budget=10,
    )
    env = make_env(cfg)
    policy = PolicyNet()
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    ep_rewards = []
    rolling_rewards = deque(maxlen=rolling_window)
    rolling_losses = deque(maxlen=rolling_window)

    thresholds = [2, 4, 6, 8, 10]
    reached = {t: None for t in thresholds}

    # CSV logging
    returns_csv = os.path.join(out_dir, "train_returns.csv")
    losses_csv = os.path.join(out_dir, "losses.csv")
    with open(returns_csv, "w", newline="") as fcsv:
        csv.writer(fcsv).writerow(["episode", "return", f"rolling_mean_return_{rolling_window}", "step"])
    with open(losses_csv, "w", newline="") as flcsv:
        csv.writer(flcsv).writerow(["step", "policy_loss", "value_loss", "entropy"])

    acts_episode = []
    agree_flags = []
    turns_used = []
    env_steps = 0
    for ep in trange(total_episodes, desc="REINFORCE"):
        obs, info = env.reset(seed=seed + ep)
        traj_logps = []
        traj_rewards = []
        steps = 0

        done = False
        while not done:
            action, logp = policy.sample(
                obs,
                info.get("oA_max", [4, 4, 4]),
                action_mask=info.get("action_mask"),
                disable_end_above_turns=max(0, max_turns - 2),
            )
            obs, reward, done, truncated, info = env.step(action)
            traj_logps.append(logp)
            traj_rewards.append(float(reward))
            steps += 1
            env_steps += 1
            if truncated:
                break
            if int(total_env_steps) > 0 and env_steps >= int(total_env_steps):
                done = True

        # Compute returns
        G = 0.0
        returns = []
        for r in reversed(traj_rewards):
            G = r + gamma * G
            returns.append(G)
        returns = list(reversed(returns))
        returns_t = torch.tensor(returns, dtype=torch.float32)
        logps_t = torch.stack(traj_logps)

        loss = -(logps_t * (returns_t - returns_t.mean())).sum() - entropy_coef * logps_t.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ep_ret = sum(traj_rewards)
        ep_rewards.append(ep_ret)
        rolling_rewards.append(ep_ret)
        rolling_losses.append(float(loss.item()))

        # thresholds
        mean_r = np.mean(rolling_rewards) if len(rolling_rewards) > 0 else ep_ret
        for t in thresholds:
            if reached[t] is None and mean_r >= t:
                reached[t] = ep + 1

        tb.add_scalar("reward/episode_return", ep_ret, ep + 1)
        tb.add_scalar("reward/rolling_mean", mean_r, ep + 1)
        tb.add_scalar("loss/policy_loss", float(loss.item()), ep + 1)
        # append csv row each episode
        with open(returns_csv, "a", newline="") as fcsv:
            csv.writer(fcsv).writerow([ep + 1, ep_ret, float(mean_r), ep + 1])
        # losses per update
        with open(losses_csv, "a", newline="") as flcsv:
            csv.writer(flcsv).writerow([ep + 1, float(loss.item()), float("nan"), float(0.0)])

        # episode summaries for final stats
        acts_episode.append(int(action["act_type"]))
        agree_flags.append(bool(env.last_partner_act == env.ACT_AGREE or action["act_type"] == env.ACT_AGREE))
        turns_used.append(steps)

        if save_every_episodes > 0 and (ep + 1) % save_every_episodes == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_ep_{ep+1}.pt")
            torch.save({"policy": policy.state_dict()}, ckpt_path)
            metrics = {
                "episode": ep + 1,
                "rolling_mean_reward": float(mean_r),
                "rolling_mean_loss": float(np.mean(rolling_losses) if len(rolling_losses) else 0.0),
                "threshold_episodes": reached,
            }
            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

    # final save
    torch.save({"policy": policy.state_dict()}, os.path.join(ckpt_dir, "ckpt_final.pt"))
    with open(os.path.join(out_dir, "status.json"), "w") as f:
        json.dump({"status": "OK"}, f, indent=2)

    # final stats JSON/CSV
    from evaluation.metrics import episode_metrics  # type: ignore
    mets = episode_metrics(ep_rewards, acts_episode, turns_used, agree_flags)
    final = {
        **mets,
        "episodes_trained": int(len(ep_rewards)),
        "seed": int(seed),
        "algo": "reinforce",
        "hint_mode": "none",
        "k": 0,
        "hint_provider": "none",
        "failure_ratio": 0.0,
    }
    with open(os.path.join(out_dir, "final_stats.json"), "w") as f:
        json.dump(final, f, indent=2)
    with open(os.path.join(out_dir, "final_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(final.keys()))
        w.writeheader(); w.writerow(final)
    # Plot rolling mean reward
    try:
        # Reload returns CSV for plotting
        xs, ys = [], []
        with open(returns_csv, "r") as fcsv:
            r = csv.DictReader(fcsv)
            for row in r:
                xs.append(int(row["episode"]))
                ys.append(float(row.get(f"rolling_mean_return_{rolling_window}", 0.0)))
        plt.figure(figsize=(6,4))
        plt.plot(xs, ys, label=f"Rolling mean return ({rolling_window})")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rolling_reward.png"), dpi=150)
    except Exception:
        pass
    # Plot training loss curve (policy_loss over steps)
    try:
        xs_l, ys_l = [], []
        with open(losses_csv, "r") as flcsv:
            r = csv.DictReader(flcsv)
            for row in r:
                step_val = row.get("step") or row.get("episode") or row.get("update")
                if step_val is None:
                    continue
                xs_l.append(int(step_val))
                ys_l.append(float(row.get("policy_loss", 0.0)))
        if len(xs_l) > 0:
            plt.figure(figsize=(6,4))
            plt.plot(xs_l, ys_l, label="policy_loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "training_loss.png"), dpi=150)
    except Exception:
        pass
    tb.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--total_episodes", type=int, default=2000)
    p.add_argument("--total_env_steps", type=int, default=0, help="Stop after this many env steps if >0")
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max_turns", type=int, default=10)
    p.add_argument("--rolling_window", type=int, default=100)
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--save_every_episodes", type=int, default=100)
    args = p.parse_args()
    train(
        out_dir=args.out_dir,
        total_episodes=args.total_episodes,
        seed=args.seed,
        max_turns=args.max_turns,
        rolling_window=args.rolling_window,
        save_every_episodes=args.save_every_episodes,
        total_env_steps=args.total_env_steps,
        entropy_coef=args.entropy_coef,
        num_envs=args.num_envs,
    )
