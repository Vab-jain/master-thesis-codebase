#!/usr/bin/env python3
import os
import json
from typing import List, Optional
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from datasets import load_dataset
from deal_or_no_deal_env import register_deal_or_no_deal
from deal_or_no_deal_env.env import NegotiationConfig
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import csv


class ExpertNet(nn.Module):
    def __init__(self, hidden=128, dropout_p: float = 0.2):
        super().__init__()
        # Input: counts(3), my_utils(3), last_partner_act(1), last_partner_offer_for_me(3), turns_remaining(1) => 11 dims
        self.backbone = nn.Sequential(
            nn.Linear(11, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Act type head (5-way) and three oA heads (each [0..4])
        self.act_head = nn.Linear(hidden, 5)
        self.heads = nn.ModuleList([nn.Linear(hidden, 5) for _ in range(3)])

    def _encode(self, counts, utils, last_act, last_offer, turns):
        return torch.cat([counts, utils, last_act, last_offer, turns], dim=-1)

    def forward(self, counts, utils, last_act=None, last_offer=None, turns=None):
        # Default zeros for missing contextual fields (training from dialogues)
        B = counts.shape[0]
        if last_act is None:
            last_act = torch.zeros((B, 1), dtype=counts.dtype, device=counts.device)
        if last_offer is None:
            last_offer = torch.zeros((B, 3), dtype=counts.dtype, device=counts.device)
        if turns is None:
            turns = torch.zeros((B, 1), dtype=counts.dtype, device=counts.device)
        x = self._encode(counts, utils, last_act, last_offer, turns)
        h = self.backbone(x)
        act_logits = self.act_head(h)
        oA_logits = [head(h) for head in self.heads]
        return act_logits, oA_logits


def parse_output_to_oA(output: str) -> Optional[List[int]]:
    # Extract first triple item0=K item1=K item2=K; ignore other tokens like <disagree>
    matches = re.findall(r"item([012])=(\d+)", output)
    if not matches:
        return None
    # Build order 0,1,2 from first occurrence
    result = [None, None, None]
    for idx_str, val_str in matches:
        idx = int(idx_str)
        if result[idx] is None:
            result[idx] = int(val_str)
        if all(v is not None for v in result):
            break
    if any(v is None for v in result):
        return None
    return [int(result[0]), int(result[1]), int(result[2])]


def train(dataset_script_path="./deal_or_no_dialog/deal_or_no_dialog.py",
           out_dir="runs/expert_sl_seed412_30ep_bs64_hidden64", 
           epochs=60, lr=3e-4, 
           batch_size=64, 
           seed=412, 
           eval_episodes=100,
           env_max_turns: int = 20,
           patience: int = 8,
           min_delta: float = 1e-3,
           min_normalized_reward: Optional[float] = None):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tb = SummaryWriter(log_dir=out_dir)

    ds = load_dataset(dataset_script_path, name="dialogues")
    train_ds = ds["train"]
    val_ds = ds["validation"]

    def collate(batch):
        valid = []
        for ex in batch:
            parsed = parse_output_to_oA(ex["output"])
            if parsed is None:
                continue
            counts_i = [int(x) for x in ex["input"]["count"]]
            utils_i = [int(x) for x in ex["input"]["value"]]
            oA_i = [int(x) for x in parsed]
            # Clamp oA to feasible [0, counts]
            oA_i = [max(0, min(oA_i[j], counts_i[j])) for j in range(3)]
            # Label orientation correction: prefer agent-utility maximizing side
            u_dot_oA = sum(utils_i[j] * oA_i[j] for j in range(3))
            u_dot_comp = sum(utils_i[j] * (counts_i[j] - oA_i[j]) for j in range(3))
            if u_dot_comp > u_dot_oA:
                oA_i = [counts_i[j] - oA_i[j] for j in range(3)]
            # Optional: filter by normalized reward quality
            if min_normalized_reward is not None:
                max_val = sum(utils_i[j] * counts_i[j] for j in range(3))
                cur_val = sum(utils_i[j] * oA_i[j] for j in range(3))
                norm = (cur_val / max_val) if max_val > 0 else 0.0
                if norm < float(min_normalized_reward):
                    continue
            valid.append((counts_i, utils_i, oA_i))
        if not valid:
            return None
        counts = torch.tensor([v[0] for v in valid], dtype=torch.float32)
        utils = torch.tensor([v[1] for v in valid], dtype=torch.float32)
        oA = torch.tensor([v[2] for v in valid], dtype=torch.long)
        # Clamp to [0,4] just in case
        counts = torch.clamp(counts, 0, 4)
        oA = torch.clamp(oA, 0, 4)
        # Normalize inputs to fixed scales to match env ranges
        counts = counts / 4.0
        utils = utils / 10.0
        # Dialogue data lacks dynamics; set contextual inputs to zeros
        last_act = torch.zeros((counts.shape[0], 1), dtype=torch.float32)
        last_offer = torch.zeros((counts.shape[0], 3), dtype=torch.float32)
        turns = torch.zeros((counts.shape[0], 1), dtype=torch.float32)
        # Act type target: propose (0) by default; agree if label is close to offered (not available here) -> keep 0
        act_type = torch.zeros((counts.shape[0],), dtype=torch.long)
        return counts, utils, last_act, last_offer, turns, oA, act_type

    model = ExpertNet()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=1e-5)

    def run_epoch(split_ds, train_mode=True, show_progress=False):
        model.train(train_mode)
        total_loss = 0.0
        n = 0
        # simple batching
        for i in trange(0, len(split_ds), batch_size, disable=not show_progress):
            batch = [split_ds[j] for j in range(i, min(i + batch_size, len(split_ds)))]
            packed = collate(batch)
            if packed is None:
                continue
            counts, utils, last_act, last_offer, turns, oA, act_type_t = packed
            act_logits, o_heads = model(counts, utils, last_act, last_offer, turns)
            # Multi-task loss: act_type CE + masked oA losses only when act_type ∈ {0,1}
            loss_act = ce(act_logits, act_type_t)
            mask_o = (act_type_t == 0) | (act_type_t == 1)
            if mask_o.any():
                idx = mask_o.nonzero(as_tuple=False).squeeze(-1)
                loss_o = sum(ce(o_heads[k][idx], oA[idx, k]) for k in range(3))
            else:
                loss_o = torch.tensor(0.0, dtype=torch.float32)
            loss = loss_act + loss_o
            if train_mode:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            total_loss += float(loss.item()) * len(batch)
            n += len(batch)
        return total_loss / max(1, n)

    best_val = float("inf")
    epochs_without_improve = 0
    # training log CSV
    log_csv = os.path.join(out_dir, "train_log.csv")
    with open(log_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    for ep in trange(1, epochs + 1, desc="Training epochs"):
        train_loss = run_epoch(train_ds, train_mode=True)
        val_loss = run_epoch(val_ds, train_mode=False)
        metrics = {"epoch": ep, "train_loss": train_loss, "val_loss": val_loss}
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        tb.add_scalar("loss/train", train_loss, ep)
        tb.add_scalar("loss/val", val_loss, ep)
        tb.add_scalar("lr", float(opt.param_groups[0]["lr"]), ep)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"expert_ep{ep}.pt"))
        if val_loss + min_delta < best_val:
            best_val = val_loss
            epochs_without_improve = 0
            # Save best both in checkpoints subfolder and at root for compatibility with downstream consumers
            best_ckpt_sub = os.path.join(ckpt_dir, "expert_best.pt")
            best_ckpt_root = os.path.join(out_dir, "expert_best.pt")
            torch.save(model.state_dict(), best_ckpt_sub)
            torch.save(model.state_dict(), best_ckpt_root)
        else:
            epochs_without_improve += 1
        with open(log_csv, "a", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([ep, float(train_loss), float(val_loss)])
        scheduler.step()
        if epochs_without_improve >= patience:
            break

    # Evaluate expert policy inside the environment by greedy argmax over heads to propose oA
    cfg = NegotiationConfig(
        use_dataset=True,
        dataset_script_path=dataset_script_path,
        dataset_config_name="dialogues",
        max_turns=int(env_max_turns),
        normalize_utilities_to_max_points=False,
        max_points_budget=10,
    )
    import gymnasium as gym
    env_id = register_deal_or_no_deal()
    env = gym.make(env_id, config=cfg)
    model.eval()
    def act(obs, info, step_idx):
        counts = torch.from_numpy(np.array(obs["counts"], dtype=np.float32) / 4.0).unsqueeze(0)
        utils = torch.from_numpy(np.array(obs["my_utilities"], dtype=np.float32) / 10.0).unsqueeze(0)
        last_act = torch.tensor([[float(obs.get("last_partner_act", 0))]], dtype=torch.float32)
        last_offer = torch.from_numpy(np.array(obs["last_partner_offer_for_me"], dtype=np.float32) / 4.0).unsqueeze(0)
        turns = torch.tensor([[float(obs.get("turns_remaining", 0))]], dtype=torch.float32) / 10.0
        act_logits, o_heads = model(counts, utils, last_act, last_offer, turns)
        act_type = int(torch.argmax(act_logits, dim=-1).item())
        # Mask END early
        ACT_PROPOSE, ACT_INSIST, ACT_AGREE, ACT_DISAGREE, ACT_END = 0, 1, 2, 3, 4
        mask = info.get("action_mask")
        if mask is not None and obs.get("turns_remaining", 0) > 2:
            if act_type == ACT_END:
                act_type = ACT_PROPOSE if mask[ACT_PROPOSE] else (ACT_INSIST if mask[ACT_INSIST] else ACT_DISAGREE)
        # If partner offered and agree is allowed, check threshold
        if mask is not None and mask[ACT_AGREE] == 1:
            offered = np.array(obs["last_partner_offer_for_me"], dtype=int)
            my_u = np.array(obs["my_utilities"], dtype=int)
            val = float(np.dot(my_u, offered))
            max_val = float(np.dot(my_u, np.array(obs["counts"], dtype=int)))
            if max_val > 0 and (val / max_val) >= 0.5:
                return {"act_type": ACT_AGREE, "oA": [0, 0, 0]}
        # Predict oA if proposing/insisting
        oA = [int(torch.argmax(o_heads[k], dim=-1).item()) for k in range(3)]
        # clamp to feasible
        oA = np.clip(np.array(oA, dtype=int), 0, np.array(obs["counts"], dtype=int)).tolist()
        if act_type not in (ACT_PROPOSE, ACT_INSIST, ACT_DISAGREE, ACT_END):
            act_type = ACT_PROPOSE
        # ensure act_type is valid
        if mask is not None and mask[act_type] == 0:
            # fallback to any valid act excluding END
            candidates = [i for i in [ACT_PROPOSE, ACT_INSIST, ACT_DISAGREE] if mask[i] == 1]
            act_type = candidates[0] if candidates else ACT_DISAGREE
        return {"act_type": act_type, "oA": oA}

    rewards = []
    xs, ys = [], []
    csv_path = os.path.join(out_dir, "eval_episodes.csv")
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["episode", "return", "length", "rolling_mean_reward"])
    window = 100
    from collections import deque
    rw = deque(maxlen=window)
    for ep in range(1, eval_episodes + 1):
        obs, info = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done:
            action = act(obs, info, steps)
            obs, reward, done, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
            if truncated:
                break
        rewards.append(total)
        rw.append(total)
        xs.append(ep)
        ys.append(float(np.mean(rw)))
        tb.add_scalar("eval/episode_return", total, ep)
        tb.add_scalar("eval/rolling_mean", float(np.mean(rw)), ep)
        with open(csv_path, "a", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([ep, total, steps, float(np.mean(rw))])
    with open(os.path.join(out_dir, "eval_metrics.json"), "w") as f:
        json.dump({"mean_reward": float(np.mean(rewards)), "median_reward": float(np.median(rewards))}, f, indent=2)
    # Plot rolling mean
    try:
        plt.figure(figsize=(6,4))
        plt.plot(xs, ys, label=f"Rolling mean reward ({window})")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rolling_reward.png"), dpi=150)
    except Exception:
        pass
    # Plot training loss curves from train_log.csv
    try:
        xs_e, tr_l, va_l = [], [], []
        with open(os.path.join(out_dir, "train_log.csv"), "r") as fcsv:
            r = csv.DictReader(fcsv)
            for row in r:
                xs_e.append(int(row.get("epoch", "0")))
                tr_l.append(float(row.get("train_loss", 0.0)))
                va_l.append(float(row.get("val_loss", 0.0)))
        if len(xs_e) > 0:
            plt.figure(figsize=(6,4))
            plt.plot(xs_e, tr_l, label="train_loss")
            plt.plot(xs_e, va_l, label="val_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "training_loss.png"), dpi=150)
    except Exception:
        pass
    tb.close()


if __name__ == "__main__":
    train()
