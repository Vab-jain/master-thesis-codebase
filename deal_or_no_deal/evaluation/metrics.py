"""Evaluation metrics stubs for RL + LLM-Hints (scaffold)."""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
from pathlib import Path


def replot_rolling_from_csv(csv_path: str, out_png: str, window: int | None = None) -> None:
    import csv
    import matplotlib.pyplot as plt  # type: ignore

    xs: List[int] = []
    ys: List[float] = []
    with Path(csv_path).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if "episode" in row and row.get("rolling_mean_reward"):
                xs.append(int(row["episode"]))
                ys.append(float(row["rolling_mean_reward"]))
    if window is not None and window > 1:
        # recompute rolling mean over full series
        import collections

        dq = collections.deque(maxlen=window)
        ys2: List[float] = []
        for v in ys:
            dq.append(v)
            ys2.append(float(sum(dq) / len(dq)))
        ys = ys2
    plt.figure(figsize=(6, 3))
    plt.plot(xs, ys)
    plt.xlabel("Episode")
    plt.ylabel("Rolling mean")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def summarize_rewards(rewards: List[float]) -> Dict[str, Any]:
    if not rewards:
        return {"mean": 0.0, "episodes": 0}
    mean_val = sum(rewards) / float(len(rewards))
    return {"mean": mean_val, "episodes": len(rewards)}


def episode_metrics(returns: List[float], acts: List[int], turns_used: List[int], agree_flags: List[bool]) -> Dict[str, Any]:
    arr = np.array(returns, dtype=float) if returns else np.zeros(1)
    avg = float(np.mean(arr)) if arr.size else 0.0
    std = float(np.std(arr)) if arr.size else 0.0
    med = float(np.median(arr)) if arr.size else 0.0
    agree_rate = float(np.mean(np.array(agree_flags, dtype=float))) if agree_flags else 0.0
    end_rate = float(np.mean(np.array([a == 4 for a in acts], dtype=float))) if acts else 0.0
    avg_turns_used = float(np.mean(np.array(turns_used, dtype=float))) if turns_used else 0.0
    return {
        "avg_return": avg,
        "std_return": std,
        "median_return": med,
        "agree_rate": agree_rate,
        "end_rate": end_rate,
        "avg_turns_used": avg_turns_used,
    }


def regret_vs_upper_bound(returns: List[float], upper_bounds: List[float]) -> float:
    if not returns or not upper_bounds:
        return 0.0
    diff = np.array(upper_bounds, dtype=float) - np.array(returns, dtype=float)
    return float(np.mean(np.maximum(0.0, diff)))


def aggregate_learning_curves(run_dirs: List[str]) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Read train_returns.csv from multiple run dirs and return (x, mean, std)."""
    import csv

    curves: List[List[float]] = []
    max_len = 0
    for d in run_dirs:
        path = Path(d) / "train_returns.csv"
        if not path.exists():
            continue
        ys: List[float] = []
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                ys.append(float(row.get("rolling_mean_reward", 0.0)))
        if ys:
            curves.append(ys)
            max_len = max(max_len, len(ys))
    if not curves:
        return ([], np.array([]), np.array([]))
    # align by truncation to min length
    min_len = min(len(c) for c in curves)
    curves = [c[:min_len] for c in curves]
    arr = np.array(curves, dtype=float)
    xs = list(range(1, min_len + 1))
    return xs, np.mean(arr, axis=0), np.std(arr, axis=0)


def plot_learning_curves_avg(method_to_runs: Dict[str, List[str]], out_png: str) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np

    plt.figure(figsize=(8, 4))
    for method, runs in method_to_runs.items():
        xs, mean, std = aggregate_learning_curves(runs)
        if len(xs) == 0:
            continue
        mean = np.array(mean)
        std = np.array(std)
        plt.plot(xs, mean, label=method)
        plt.fill_between(xs, mean - std, mean + std, alpha=0.2)
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Rolling mean")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_final_bar_avg(method_to_eval_csvs: Dict[str, List[str]], out_png: str) -> None:
    import csv
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np

    methods: List[str] = []
    means: List[float] = []
    stds: List[float] = []
    for method, files in method_to_eval_csvs.items():
        vals: List[float] = []
        for f in files:
            path = Path(f)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                r = csv.DictReader(fh)
                try:
                    row = next(r)
                    vals.append(float(row.get("avg_return", 0.0)))
                except StopIteration:
                    pass
        if vals:
            methods.append(method)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
    if not methods:
        return

    x = np.arange(len(methods))
    plt.figure(figsize=(8, 4))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, methods, rotation=30, ha="right")
    plt.ylabel("Avg Return")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_variants_for_algo(exp_root: str, algo: str, k: int, out_png: str) -> None:
    """Aggregate rolling curves for variants of a given algo and k.

    Expects directory layout: runs/EXP_NAME/<algo>/<variant>/seed_*/train_returns.csv
    Variants included:
      - RL
      - RL-llm_k{K}
      - RL-llm_k{K}_random
      - RL-llm_k{K}_expert
    """
    import glob
    import csv
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np

    methods = [
        "RL",
        f"RL-llm_k{k}",
        f"RL-llm_k{k}_random",
        f"RL-llm_k{k}_expert",
    ]
    plt.figure(figsize=(8, 4))
    for m in methods:
        run_dirs = sorted(glob.glob(f"{exp_root}/{algo}/{m}/seed_*/"))
        xs, mean, std = aggregate_learning_curves(run_dirs)
        if len(xs) == 0:
            continue
        mean = np.array(mean); std = np.array(std)
        plt.plot(xs, mean, label=m)
        plt.fill_between(xs, mean - std, mean + std, alpha=0.2)
    plt.legend(); plt.xlabel("Episode"); plt.ylabel("Rolling mean")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png); plt.close()


def plot_k1_vs_k3(exp_root: str, algo: str, out_png: str) -> None:
    import glob
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np

    methods = ["RL", "RL-llm_k1", "RL-llm_k3"]
    plt.figure(figsize=(8, 4))
    for m in methods:
        run_dirs = sorted(glob.glob(f"{exp_root}/{algo}/{m}/seed_*/"))
        xs, mean, std = aggregate_learning_curves(run_dirs)
        if len(xs) == 0:
            continue
        mean = np.array(mean)
        std = np.array(std)
        plt.plot(xs, mean, label=m)
        plt.fill_between(xs, mean - std, mean + std, alpha=0.2)
    plt.legend(); plt.xlabel("Episode"); plt.ylabel("Rolling mean")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png); plt.close()


def plot_cross_algo(exp_root: str, out_png: str) -> None:
    """Bar chart comparing PPO vs REINFORCE vs best LLM variants by eval returns.

    Finds best-k by averaging eval_metrics.csv across seeds for k in {1,3}.
    """
    import glob
    import csv
    import numpy as np
    import matplotlib.pyplot as plt  # type: ignore

    def avg_eval(method_glob: str) -> float:
        vals: List[float] = []
        for f in glob.glob(method_glob):
            try:
                with Path(f).open("r", encoding="utf-8") as fh:
                    r = csv.DictReader(fh)
                    row = next(r)
                    vals.append(float(row.get("avg_return", 0.0)))
            except Exception:
                continue
        return float(np.mean(vals)) if vals else 0.0

    ppo_base = avg_eval(f"{exp_root}/ppo/RL/seed_*/eval_metrics.csv")
    reinf_base = avg_eval(f"{exp_root}/reinforce/RL/seed_*/eval_metrics.csv")
    ppo_k1 = avg_eval(f"{exp_root}/ppo/RL-llm_k1/seed_*/eval_metrics.csv")
    ppo_k3 = avg_eval(f"{exp_root}/ppo/RL-llm_k3/seed_*/eval_metrics.csv")
    reinf_k1 = avg_eval(f"{exp_root}/reinforce/RL-llm_k1/seed_*/eval_metrics.csv")
    reinf_k3 = avg_eval(f"{exp_root}/reinforce/RL-llm_k3/seed_*/eval_metrics.csv")
    ppo_best = max(ppo_k1, ppo_k3)
    reinf_best = max(reinf_k1, reinf_k3)

    methods = ["PPO", "REINFORCE", "PPO_LLM_best_k", "REINFORCE_LLM_best_k"]
    means = [ppo_base, reinf_base, ppo_best, reinf_best]
    x = np.arange(len(methods))
    plt.figure(figsize=(8, 4))
    plt.bar(x, means)
    plt.xticks(x, methods, rotation=15)
    plt.ylabel("Avg Eval Return")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png); plt.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--replot_csv", type=str, default=None)
    p.add_argument("--replot_out", type=str, default=None)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--curves_out", type=str, default=None)
    p.add_argument("--final_out", type=str, default=None)
    args = p.parse_args()
    if args.replot_csv and args.replot_out:
        replot_rolling_from_csv(args.replot_csv, args.replot_out, args.window)


