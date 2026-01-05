from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm  # type: ignore

from train.common.nn import build_policy


EnvFactory = Callable[[], Any]


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


def train(config: Dict[str, Any], env_fn: EnvFactory, hint_adapter: Any, logger: Any, eval_env_fn: EnvFactory | None = None) -> None:
    training = config.get("training", {})
    ppo_cfg = training.get("ppo", {})
    hint_dim = int(config.get("hints", {}).get("feature_dim", 0))

    env = env_fn()
    obs0, info0 = env.reset(seed=int(config.get("seed", 42)))
    input_dim = hint_adapter.input_dim_from_obs(obs0, info0, hint_dim)
    policy = build_policy(input_dim=input_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(ppo_cfg.get("learning_rate", 3e-4)))

    gamma = float(ppo_cfg.get("gamma", 0.99))
    gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
    clip_coef = float(ppo_cfg.get("clip_coef", 0.2))
    update_epochs = int(ppo_cfg.get("update_epochs", 4))
    minibatch_size = int(ppo_cfg.get("minibatch_size", 64))
    ent_coef = float(ppo_cfg.get("ent_coef", 0.01))
    vf_coef = float(ppo_cfg.get("vf_coef", 0.5))
    max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))
    rollout_steps = int(ppo_cfg.get("rollout_steps", 128))

    # Buffers
    obs_buf: List[np.ndarray] = []
    info_buf: List[Dict[str, Any]] = []
    actions_buf: List[Dict[str, Any]] = []
    logprobs_buf: List[float] = []
    values_buf: List[float] = []
    rewards_buf: List[float] = []
    dones_buf: List[bool] = []

    episode_return = 0.0
    episode_loss = 0.0
    episodes_done = 0
    update_idx = 0

    returns_history: List[float] = []

    obs, info = obs0, info0
    # Debug: log input dim vs base dim once at start (for --hint random acceptance)
    try:
        base_dim = hint_adapter.obs_to_base_only(obs).shape[0]
        aug_dim = hint_adapter.obs_to_input(obs, info).shape[0]
        print(f"[PPO] obs dims: base=({base_dim},) -> with_hint=({aug_dim},)")
    except Exception:
        pass
    steps_cap = int(training.get("num_train_steps", training.get("total_env_steps", 0)))
    total_episodes = int(training.get("num_train_episodes", 2**31-1))
    env_steps = 0

    # Progress bars
    pbar_steps = tqdm(total=steps_cap, desc="PPO", unit="step") if steps_cap > 0 else None
    pbar_eps = None if steps_cap > 0 else tqdm(total=total_episodes, desc="PPO", unit="ep")

    while (steps_cap <= 0 or env_steps < steps_cap) and episodes_done < total_episodes:
        # Rollout
        for _ in range(rollout_steps):
            x_np = hint_adapter.obs_to_input(obs, info)
            x = torch.from_numpy(x_np).unsqueeze(0)
            out = policy(x)

            # Action masking
            act_mask = hint_adapter.get_action_mask(info)
            action_logits = hint_adapter.apply_action_mask(out["action_logits"], act_mask)
            dist_act = Categorical(logits=action_logits)
            act_idx = int(dist_act.sample().item())
            logp_act = float(dist_act.log_prob(torch.tensor([act_idx])).item())

            # oA heads with caps
            oA_max = hint_adapter.get_oa_caps(info)
            oA_vals: List[int] = []
            logp_oa = 0.0
            for i in range(3):
                logits_i = hint_adapter.apply_oa_cap(out[f"oA_logits_{i}"], int(oA_max[i]))
                dist_i = Categorical(logits=logits_i)
                idx_i = int(dist_i.sample().item())
                oA_vals.append(idx_i)
                if act_idx in (0, 1):
                    logp_oa += float(dist_i.log_prob(torch.tensor([idx_i])).item())

            value = float(out["value"].item())

            action_dict = {"act_type": act_idx}
            if act_idx in (0, 1):
                action_dict["oA"] = oA_vals

            next_obs, reward, terminated, truncated, next_info = env.step(action_dict)
            env_steps += 1
            if pbar_steps is not None:
                pbar_steps.update(1)

            obs_buf.append(x_np)
            info_buf.append(info)
            actions_buf.append(action_dict)
            logprobs_buf.append(logp_act + logp_oa)
            values_buf.append(value)
            rewards_buf.append(float(reward))
            done = bool(terminated or truncated)
            dones_buf.append(done)

            episode_return += float(reward)
            if done:
                episodes_done += 1
                returns_history.append(episode_return)
                logger.log_episode(episodes_done, env_steps, loss=episode_loss, return_=episode_return)
                if pbar_eps is not None:
                    pbar_eps.update(1)
                episode_return = 0.0
                episode_loss = 0.0
                next_obs, next_info = env.reset()

            if (steps_cap > 0 and env_steps >= steps_cap) or episodes_done >= total_episodes:
                break

            obs, info = next_obs, next_info

        if (steps_cap > 0 and env_steps >= steps_cap) or episodes_done >= total_episodes:
            break

        # GAE and update
        adv, rets = compute_gae(rewards_buf, dones_buf, values_buf, gamma, gae_lambda)
        adv_t = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8))
        rets_t = torch.from_numpy(rets)
        obs_t = torch.from_numpy(np.stack(obs_buf, axis=0))
        old_logprobs_t = torch.tensor(logprobs_buf)

        bsz_total = obs_t.shape[0]
        idxs = np.arange(bsz_total)
        mb = max(1, min(minibatch_size, bsz_total))

        def compute_lp_ent(indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            out_b = policy(obs_t[indices])
            lp_new = torch.zeros(len(indices))
            ent_b = torch.zeros(len(indices))
            for j, t_idx in enumerate(indices):
                info_t = info_buf[int(t_idx)]
                obsd = actions_buf[int(t_idx)]
                act_taken = int(obsd["act_type"]) if "act_type" in obsd else int(obsd.get("act", 0))
                act_logits_t = hint_adapter.apply_action_mask(out_b["action_logits"][j], hint_adapter.get_action_mask(info_t))
                dist_t = Categorical(logits=act_logits_t)
                lp = dist_t.log_prob(torch.tensor(act_taken))
                ent = dist_t.entropy()
                if act_taken in (0, 1) and "oA" in obsd:
                    oA_max_l = hint_adapter.get_oa_caps(info_t)
                    for i in range(3):
                        logits_i = hint_adapter.apply_oa_cap(out_b[f"oA_logits_{i}"][j], int(oA_max_l[i]))
                        dist_i = Categorical(logits=logits_i)
                        lp = lp + dist_i.log_prob(torch.tensor(int(obsd["oA"][i])))
                        ent = ent + dist_i.entropy()
                lp_new[j] = lp
                ent_b[j] = ent
            return lp_new, ent_b, out_b["value"]

        for _ in range(int(update_epochs)):
            np.random.shuffle(idxs)
            for start in range(0, bsz_total, mb):
                mb_idx = idxs[start : min(start + mb, bsz_total)]
                lp_new_mb, ent_mb, values_mb = compute_lp_ent(mb_idx)
                ratio = (lp_new_mb - old_logprobs_t[mb_idx]).exp()
                adv_mb = adv_t[mb_idx]
                pg1 = ratio * adv_mb
                pg2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv_mb
                policy_loss = -torch.min(pg1, pg2).mean()

                value_loss = 0.5 * (rets_t[mb_idx] - values_mb).pow(2).mean()
                entropy_loss = -ent_mb.mean()
                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

                episode_loss = float(loss.item())

        logger.log_update(update_idx, policy_loss=float(policy_loss.item()), value_loss=float(value_loss.item()), entropy=float(-entropy_loss.item()), steps=env_steps)
        update_idx += 1

        # Clear buffers
        obs_buf.clear(); info_buf.clear(); actions_buf.clear(); logprobs_buf.clear(); values_buf.clear(); rewards_buf.clear(); dones_buf.clear()

    # Close bars
    try:
        if pbar_steps is not None:
            pbar_steps.close()
        if pbar_eps is not None:
            pbar_eps.close()
    except Exception:
        pass

    # Minimal end-of-training print
    print(f"[PPO] training complete: steps={env_steps} episodes={episodes_done} out_dir={logger.out_dir}")


