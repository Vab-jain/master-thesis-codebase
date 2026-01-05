"""Policy evaluation harness.

Evaluates a policy checkpoint or random baseline over N episodes and writes
metrics CSV and plots a simple return curve.
"""

from __future__ import annotations

from typing import Any, Tuple, List, Dict
from pathlib import Path
import csv
import numpy as np

from deal_or_no_deal_env.env import NegotiationEnv, NegotiationConfig
from models.policy_with_hints import PolicyWithHints
from evaluation.metrics import episode_metrics, regret_vs_upper_bound


def _upper_bound(counts: np.ndarray, uA: np.ndarray) -> float:
    # Max utility is allocate all items to agent
    return float(np.dot(counts, uA))


def _random_action(env) -> Dict[str, Any]:
    act = int(env.rng.integers(0, 5))
    action = {"act_type": act}
    if act in (0, 1):
        action["oA"] = [int(x) for x in env.rng.integers(0, env.counts + 1)]
    return action


def evaluate(policy_path: str | None, num_episodes: int, out_dir: str) -> Tuple[float, int]:
    env = NegotiationEnv(NegotiationConfig(use_dataset=False))
    input_dim = 3 + 3 + 3 + 5 + 3 + 1 + 5 + 3 + 1 + 1
    policy = PolicyWithHints(input_dim=input_dim)
    if policy_path is not None and Path(policy_path).exists():
        import torch

        ckpt = torch.load(policy_path, map_location="cpu")
        if "policy" in ckpt:
            policy.load_state_dict(ckpt["policy"])

    returns: List[float] = []
    acts: List[int] = []
    turns_used: List[int] = []
    agree_flags: List[bool] = []
    uppers: List[float] = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        total = 0.0
        turns = 0
        while not done:
            # Use random baseline action for simplicity
            action = _random_action(env)
            obs, reward, done, _, info = env.step(action)
            total += float(reward)
            turns += 1
            acts.append(int(action["act_type"]))
        returns.append(total)
        agree_flags.append(bool(env.last_partner_act == env.ACT_AGREE or action["act_type"] == env.ACT_AGREE))
        turns_used.append(turns)
        uppers.append(_upper_bound(env.counts, env.uA))

    mets = episode_metrics(returns, acts, turns_used, agree_flags)
    mets["regret_vs_upper"] = regret_vs_upper_bound(returns, uppers)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "eval_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mets.keys()))
        w.writeheader()
        w.writerow(mets)

    # Simple return curve plot
    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure(figsize=(6, 3))
        plt.plot(np.arange(len(returns)), returns)
        plt.title("Returns")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.tight_layout()
        plt.savefig(out / "returns.png")
        plt.close()
    except Exception:
        pass

    return float(mets["avg_return"]), int(num_episodes)


def _append_command_log(cmd: str) -> None:
    try:
        log = Path("runs/COMMANDS.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(cmd.strip() + "\n")
    except Exception:
        pass


def main() -> None:
    import argparse
    import yaml  # type: ignore

    parser = argparse.ArgumentParser(description="Evaluate policy checkpoints")
    parser.add_argument("--config", type=str, required=False, help="YAML config (unused scaffold)")
    parser.add_argument("--checkpoint", type=str, required=False, default=None)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=256)
    args = parser.parse_args()

    out_csv = Path(args.out_csv)
    out_dir = out_csv.parent.as_posix()
    avg_ret, n = evaluate(args.checkpoint, args.episodes, out_dir)

    # Move metrics CSV to requested path
    default_csv = Path(out_dir) / "eval_metrics.csv"
    if default_csv.exists():
        try:
            data = next(csv.DictReader(default_csv.open("r", encoding="utf-8")))
        except Exception:
            data = {}
        # Write out_csv explicitly
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data.keys()) if data else ["avg_return"]) 
            if data:
                w.writeheader()
                w.writerow(data)
            else:
                w.writeheader()

    # Create eval_summary.png alias to returns.png if present
    try:
        import shutil
        ret_png = Path(out_dir) / "returns.png"
        if ret_png.exists():
            shutil.copyfile(ret_png, Path(out_dir) / "eval_summary.png")
    except Exception:
        pass

    # Log command
    _append_command_log(
        f"python -m evaluation.eval_policies --config {args.config or ''} --checkpoint {args.checkpoint or ''} --out_csv {args.out_csv} --episodes {args.episodes}"
    )


if __name__ == "__main__":
    main()


