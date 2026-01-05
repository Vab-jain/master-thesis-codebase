from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm  # type: ignore

from train.common.nn import build_policy


EnvFactory = Callable[[], Any]


def train(config: Dict[str, Any], env_fn: EnvFactory, hint_adapter: Any, logger: Any, eval_env_fn: EnvFactory | None = None) -> None:
    training = config.get("training", {})
    steps_cap = int(training.get("num_train_steps", training.get("total_env_steps", 0)))
    total_episodes = int(training.get("num_train_episodes", 2**31-1))

    env = env_fn()
    obs0, info0 = env.reset(seed=int(config.get("seed", 42)))
    hint_dim = int(config.get("hints", {}).get("feature_dim", 0))
    input_dim = hint_adapter.input_dim_from_obs(obs0, info0, hint_dim)
    policy = build_policy(input_dim=input_dim)
    optimizer = optim.Adam(policy.parameters(), lr=float(training.get("learning_rate", 3e-4)))

    entropy_coef = float(training.get("ent_coef", 0.01))
    gamma = float(training.get("gamma", 0.99))

    episodes_done = 0
    env_steps = 0
    update_idx = 0

    pbar_steps = tqdm(total=steps_cap, desc="REINFORCE", unit="step") if steps_cap > 0 else None
    pbar_eps = None if steps_cap > 0 else tqdm(total=total_episodes, desc="REINFORCE", unit="ep")

    # Debug: log input dim vs base dim once at start
    try:
        base_dim = hint_adapter.obs_to_base_only(obs0).shape[0]
        aug_dim = hint_adapter.obs_to_input(obs0, info0).shape[0]
        print(f"[REINFORCE] obs dims: base=({base_dim},) -> with_hint=({aug_dim},)")
    except Exception:
        pass

    while (steps_cap <= 0 or env_steps < steps_cap) and episodes_done < total_episodes:
        obs, info = env.reset(seed=int(config.get("seed", 42)) + episodes_done)
        traj_logps: List[torch.Tensor] = []
        traj_rewards: List[float] = []
        steps = 0
        done = False
        while not done:
            x_np = hint_adapter.obs_to_input(obs, info)
            x = torch.from_numpy(x_np).unsqueeze(0)
            out = policy(x)

            # Action
            act_logits = hint_adapter.apply_action_mask(out["action_logits"], hint_adapter.get_action_mask(info))
            dist = torch.distributions.Categorical(logits=act_logits)
            act = int(dist.sample().item())
            logp = dist.log_prob(torch.tensor(act))

            # oA
            oA_vals = [0, 0, 0]
            if act in (0, 1):
                oA_max = hint_adapter.get_oa_caps(info)
                for i in range(3):
                    logits_i = hint_adapter.apply_oa_cap(out[f"oA_logits_{i}"], int(oA_max[i]))
                    dist_i = torch.distributions.Categorical(logits=logits_i)
                    val = int(dist_i.sample().item())
                    oA_vals[i] = val
                    logp = logp + dist_i.log_prob(torch.tensor(val))

            action = {"act_type": act, "oA": oA_vals}
            obs, reward, terminated, truncated, info = env.step(action)
            traj_logps.append(logp)
            traj_rewards.append(float(reward))
            steps += 1
            env_steps += 1
            if pbar_steps is not None:
                pbar_steps.update(1)
            done = bool(terminated or truncated) or (steps_cap > 0 and env_steps >= steps_cap)

        # REINFORCE update
        G = 0.0
        returns: List[float] = []
        for r in reversed(traj_rewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float32)
        logps_t = torch.stack(traj_logps)
        loss = -(logps_t * (returns_t - returns_t.mean())).sum() - entropy_coef * logps_t.mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        episodes_done += 1
        ep_ret = float(np.sum(traj_rewards))
        logger.log_episode(episodes_done, env_steps, loss=float(loss.item()), return_=ep_ret)
        if pbar_eps is not None:
            pbar_eps.update(1)
        # Log per-episode policy loss into losses.csv for plotting
        logger.log_update(update_idx, policy_loss=float(loss.item()), value_loss=float('nan'), entropy=float(0.0), steps=env_steps)
        update_idx += 1

    try:
        if pbar_steps is not None:
            pbar_steps.close()
        if pbar_eps is not None:
            pbar_eps.close()
    except Exception:
        pass

    print(f"[REINFORCE] training complete: steps={env_steps} episodes={episodes_done} out_dir={logger.out_dir}")


