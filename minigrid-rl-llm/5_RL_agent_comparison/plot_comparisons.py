#!/usr/bin/env python3
"""
PickUpLoc Experiment Plotter
===========================
Generates the comparison figures requested in the chat conversation for a *single* experiment
results directory that follows the structure produced by `main_rl_training.py`.

Figures produced
----------------
1. baseline_vs_baseline_text.png
2. no_text_comparisons.png               (2×3 grid)
3. with_text_comparisons.png             (2×3 grid)
4. no_text_frequency_sweeps.png          (1×3 grid)
5. with_text_frequency_sweeps.png        (1×3 grid)
6. frames_to_thresholds_no_text.png      (2×3 grid – 4-bar grouped)
7. frames_to_thresholds_with_text.png    (2×3 grid – 4-bar grouped)

Usage
-----
    python plot_comparisons.py \
        --results_dir 4_RL_agent_training/RL_Training_Results_ALL_final_PickupLoc \
        --output_dir 5_RL_agent_comparison

Dependencies: matplotlib, pandas (optional for nicer bar plots), seaborn (optional).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

########################################
# ----------  CONFIG HELPERS  ----------
########################################

def _parse_summary(path: Path) -> Optional[dict]:
    """Load a training_summary.json file and return its data dict (or None on failure)."""
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def _collect_runs(results_dir: Path) -> Dict[str, dict]:
    """Scan *results_dir* for training_summary.json files and index them by a unique label.

    Key format: f"{env}|{label}" where *label* encodes baseline/oracle/llm etc.
    """
    index: Dict[str, dict] = {}

    for summary_path in results_dir.rglob("training_summary.json"):
        summary = _parse_summary(summary_path)
        if summary is None:
            continue

        cfg = summary.get("training_config", {})
        env = summary.get("environment", "unknown")

        use_hints: bool = cfg.get("use_hints", False)
        use_text: bool = cfg.get("use_text", False)
        hint_freq: int = cfg.get("hint_frequency", 0) or 0
        hint_prob: float = float(cfg.get("hint_probability", 0)) if use_hints else 0.0
        hint_prob_label = f"{hint_prob:.1f}".rstrip("0").rstrip(".")  # 0.2 / 0.5 / 1 etc.

        # Determine flavour
        if not use_hints:
            flavour = "baseline"
        elif hint_prob == 1.0:
            flavour = "oracle"
        else:
            flavour = "llm"

        # Build human-readable label used in legends
        if flavour == "baseline":
            label = "Baseline +Text" if use_text else "Baseline"
        elif flavour == "oracle":
            base = f"Oracle f{hint_freq}"
            label = f"{base} +Text" if use_text else base
        else:  # llm
            base = f"LLM p{hint_prob_label} f{hint_freq}"
            label = f"{base} +Text" if use_text else base

        key = f"{env}|{label}"
        index[key] = {
            "meta": {
                "env": env,
                "use_text": use_text,
                "flavour": flavour,
                "hint_freq": hint_freq,
                "hint_prob": hint_prob,
                "label": label,
            },
            "summary": summary,
        }

    return index

########################################
# ----------  DATA HELPERS  -----------
########################################

def _extract_curve(summary: dict) -> Tuple[List[int], List[float], List[float]]:
    """Return (updates, mean_return, win_rate[%]) from *summary*."""
    logs = summary.get("training_logs", [])
    if not logs:
        return [], [], []
    updates = [entry["update"] for entry in logs]
    mean_ret = [entry["return_per_episode"]["mean"] for entry in logs]
    win_rate = [entry.get("rolling_win_rate", 0) * 100 for entry in logs]
    return updates, mean_ret, win_rate


def _frames_to_thresholds(summary: dict) -> Dict[str, Optional[int]]:
    """Return mapping threshold->frames from summary (None if not reached)."""
    return summary.get("success_rate_thresholds", {})

########################################
# ----------  PLOTTING CORE  ----------
########################################

# ---------------------------------------------------------------------------
# Seaborn aesthetic configuration
# ---------------------------------------------------------------------------

# A pleasant white-grid style with larger fonts for presentations
sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)

# Retrieve a reusable palette (deep = 10 distinct colours)
_PALETTE = sns.color_palette("deep")

# NOTE: We keep a COLORS dict for potential future explicit colour control, but
# default seaborn palette will usually suffice for line plots. You can still
# map specific labels to colours here if you wish.
COLORS = {label: _PALETTE[i % len(_PALETTE)] for i, label in enumerate([
    "Baseline", "Baseline +Text", "Oracle", "LLM0.2", "LLM0.5"])}


def _plot_curves(ax, runs: List[dict], metric: str = "win_rate"):
    """Plot *metric* curves for given runs on *ax* (overlay)."""
    for run in runs:
        updates, mean_ret, win_rate = _extract_curve(run["summary"])
        if not updates:
            continue
        y = win_rate if metric == "win_rate" else mean_ret
        ax.plot(updates, y, label=run["meta"]["label"], linewidth=2)
    ax.set_xlabel("Updates")
    if metric == "win_rate":
        ax.set_ylabel("Win-Rate %")
        ax.set_ylim(0, 100)
    else:
        ax.set_ylabel("Mean Return")
        ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def _plot_threshold_bars(ax, runs: List[dict], thresholds=(0.25, 0.5, 0.75, 0.9)):
    """Grouped bar chart of frames-to-threshold for each run."""
    bar_width = 0.15
    x = np.arange(len(thresholds))  # positions per threshold
    for idx, run in enumerate(runs):
        frames_map = _frames_to_thresholds(run["summary"])
        y = [frames_map.get(str(t), None) for t in thresholds]
        y = [v if v is not None else np.nan for v in y]
        color = _PALETTE[idx % len(_PALETTE)]
        ax.bar(x + idx * bar_width, y, bar_width,
               label=run["meta"]["label"], color=color, edgecolor="black", linewidth=0.7)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([f"{int(t*100)}%" for t in thresholds])
    ax.set_ylabel("Frames")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7)

########################################
# ----------  FIGURE BUILDERS  --------
########################################


def build_baseline_vs_text(index: Dict[str, dict], env: str, out_dir: Path):
    baseline = next((run for run in index.values() if run["meta"]["env"] == env and run["meta"]["flavour"] == "baseline" and not run["meta"]["use_text"]), None)
    baseline_text = next((run for run in index.values() if run["meta"]["env"] == env and run["meta"]["flavour"] == "baseline" and run["meta"]["use_text"]), None)
    if not baseline or not baseline_text:
        print("[baseline_vs_text] Required runs missing – skipping plot")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    _plot_curves(ax, [baseline, baseline_text], metric="win_rate")
    ax.set_title("Baseline vs Baseline +Text (Win-Rate)")
    fig.tight_layout()
    fp = out_dir / "baseline_vs_baseline_text.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {fp}")


def build_prob_freq_grid(index: Dict[str, dict], env: str, use_text: bool, out_dir: Path):
    """Construct 2×3 grid comparing Baseline vs Oracle vs LLM for probabilities 0.2 & 0.5."""
    title_suffix = "with_text" if use_text else "no_text"
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)

    baseline_run = next((run for run in index.values() if run["meta"]["env"] == env and run["meta"]["flavour"] == "baseline" and run["meta"]["use_text"] == use_text), None)
    if not baseline_run:
        print(f"[prob_freq_grid] Missing baseline run for use_text={use_text} – skipping")
        plt.close(fig)
        return

    for row, prob in enumerate([0.2, 0.5]):
        for col, freq in enumerate([1, 5, 10]):
            ax = axes[row, col]
            # Find other runs
            oracle_run = next((run for run in index.values() if run["meta"].values() and run["meta"]["env"] == env and run["meta"]["flavour"] == "oracle" and run["meta"]["hint_freq"] == freq and run["meta"]["use_text"] == use_text), None)
            llm_run = next((run for run in index.values() if run["meta"]["env"] == env and run["meta"]["flavour"] == "llm" and abs(run["meta"]["hint_prob"] - prob) < 1e-3 and run["meta"]["hint_freq"] == freq and run["meta"]["use_text"] == use_text), None)
            runs = [baseline_run]
            if oracle_run:
                runs.append(oracle_run)
            if llm_run:
                runs.append(llm_run)
            _plot_curves(ax, runs, metric="win_rate")
            ax.set_title(f"p={prob} f={freq}")

    fig.suptitle(f"Baseline vs Oracle vs LLM • {title_suffix}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fp = out_dir / f"{title_suffix}_comparisons.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {fp}")


def build_frequency_sweeps(index: Dict[str, dict], env: str, use_text: bool, out_dir: Path):
    suffix = "with_text" if use_text else "no_text"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    # Oracle sweep
    oracle_runs = [run for run in index.values() if run["meta"]["env"] == env and run["meta"]["flavour"] == "oracle" and run["meta"]["use_text"] == use_text]
    oracle_runs = sorted(oracle_runs, key=lambda r: r["meta"]["hint_freq"])
    _plot_curves(axes[0], oracle_runs, metric="win_rate")
    axes[0].set_title("Oracle – freq sweep")

    # LLM p0.2
    llm_p02 = [run for run in index.values() if run["meta"]["env"] == env and run["meta"]["flavour"] == "llm" and abs(run["meta"]["hint_prob"] - 0.2) < 1e-3 and run["meta"]["use_text"] == use_text]
    llm_p02 = sorted(llm_p02, key=lambda r: r["meta"]["hint_freq"])
    _plot_curves(axes[1], llm_p02, metric="win_rate")
    axes[1].set_title("LLM p0.2 – freq sweep")

    # LLM p0.5
    llm_p05 = [run for run in index.values() if run["meta"]["env"] == env and run["meta"]["flavour"] == "llm" and abs(run["meta"]["hint_prob"] - 0.5) < 1e-3 and run["meta"]["use_text"] == use_text]
    llm_p05 = sorted(llm_p05, key=lambda r: r["meta"]["hint_freq"])
    _plot_curves(axes[2], llm_p05, metric="win_rate")
    axes[2].set_title("LLM p0.5 – freq sweep")

    fig.suptitle(f"Frequency Sweeps • {suffix}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fp = out_dir / f"{suffix}_frequency_sweeps.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {fp}")


def build_threshold_grids(index: Dict[str, dict], env: str, use_text: bool, out_dir: Path):
    title_suffix = "with_text" if use_text else "no_text"
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
    baseline_run = next((run for run in index.values() if run["meta"]["env"] == env and run["meta"]["flavour"] == "baseline" and run["meta"]["use_text"] == use_text), None)
    if not baseline_run:
        print(f"[threshold_grid] Missing baseline run ({title_suffix}) – skip")
        plt.close(fig)
        return

    for row, prob in enumerate([0.2, 0.5]):
        for col, freq in enumerate([1, 5, 10]):
            ax = axes[row, col]
            oracle_run = next((run for run in index.values() if run["meta"]["flavour"] == "oracle" and run["meta"]["hint_freq"] == freq and run["meta"]["use_text"] == use_text and run["meta"]["env"] == env), None)
            llm_run = next((run for run in index.values() if run["meta"]["flavour"] == "llm" and abs(run["meta"]["hint_prob"] - prob) < 1e-3 and run["meta"]["hint_freq"] == freq and run["meta"]["use_text"] == use_text and run["meta"]["env"] == env), None)
            runs = [baseline_run]
            if oracle_run:
                runs.append(oracle_run)
            if llm_run:
                runs.append(llm_run)
            _plot_threshold_bars(ax, runs)
            ax.set_title(f"p={prob} f={freq}")

    fig.suptitle(f"Frames-to-Threshold • {title_suffix}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fp = out_dir / f"frames_to_thresholds_{title_suffix}.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {fp}")

########################################
# ----------  MAIN ROUTINE  -----------
########################################

def main():
    parser = argparse.ArgumentParser(description="Generate comparison figures for BabyAI-PickupLoc experiments.")
    parser.add_argument("--results_dir", required=True, type=Path, help="Directory containing run sub-directories with training_summary.json files")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to write figure PNGs")
    parser.add_argument("--env", nargs="*", help="Specific environment names to process (e.g., BabyAI-PickupLoc-v0 BabyAI-OpenDoor-v0). If omitted, the script processes all environments found in the results directory.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    index = _collect_runs(args.results_dir)
    if not index:
        print("❌ No runs found – check --results_dir path")
        return

    # Determine environments to process
    if args.env:
        envs = args.env
    else:
        envs = sorted({run["meta"]["env"] for run in index.values()})

    print(f"📑 Environments detected: {', '.join(envs)}")

    for env in envs:
        print(f"\n=== Processing {env} ===")

        # 1. Baseline vs Baseline+Text
        build_baseline_vs_text(index, env, args.output_dir)

        # 2. Probability × Frequency grid (no text / with text)
        build_prob_freq_grid(index, env, use_text=False, out_dir=args.output_dir)
        build_prob_freq_grid(index, env, use_text=True, out_dir=args.output_dir)

        # 3. Frequency sweeps
        build_frequency_sweeps(index, env, use_text=False, out_dir=args.output_dir)
        build_frequency_sweeps(index, env, use_text=True, out_dir=args.output_dir)

        # 4. Frames-to-threshold grids
        build_threshold_grids(index, env, use_text=False, out_dir=args.output_dir)
        build_threshold_grids(index, env, use_text=True, out_dir=args.output_dir)


if __name__ == "__main__":
    main() 