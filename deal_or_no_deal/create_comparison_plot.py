#!/usr/bin/env python3
"""
Create comparison plots (returns and/or losses) for a set of runs in a folder.

Given a base folder that contains multiple experiments of a single RL algorithm
(e.g., runs/ppo/ or runs/reinforce/), this script:
  - loads train_returns.csv and/or losses.csv from all matching subfolders
  - aligns by step index
  - plots the mean curve with variance shading across different seeds

Usage examples:
  python create_comparison_plot.py --base_dir runs/ppo --metric return --pattern ppo_random
  python create_comparison_plot.py --base_dir runs/reinforce --metric loss --pattern reinforce_random
  python create_comparison_plot.py --base_dir runs/ppo --metric both
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# Apply a consistent seaborn theme for nicer defaults
sns.set_theme(style="whitegrid", context="talk")


def _list_run_dirs(base_dir: Path, pattern: str | None) -> list[Path]:
    runs = []
    if not base_dir.exists():
        return runs
    for p in sorted(base_dir.iterdir()):
        if not p.is_dir():
            continue
        name = p.name.lower()
        if pattern is None or str(pattern).lower() in name:
            runs.append(p)
    return runs


def load_runs(base_dir: Path, pattern: str | None, file_name: str, step_col: str, value_cols: list[str]) -> dict[str, pd.DataFrame]:
    """Load CSV files from sub-runs and return a mapping of run_name -> DataFrame with selected columns.

    The DataFrame is expected to contain a step column and one or more value columns.
    """
    data: dict[str, pd.DataFrame] = {}
    for run_dir in _list_run_dirs(base_dir, pattern):
        f = run_dir / file_name
        if not f.exists():
            # skip silently to allow heterogeneous folders
            continue
        try:
            df = pd.read_csv(f)
            # select available columns
            cols = [c for c in [step_col] + value_cols if c in df.columns]
            if step_col not in cols:
                continue
            data[run_dir.name] = df[cols].copy()
        except Exception:
            continue
    return data


def aggregate_by_step(run_frames: dict[str, pd.DataFrame], step_col: str, value_col: str, rolling_window: int = 1, debug: bool = False, align: str = "intersection") -> pd.DataFrame:
    """Align on common steps across runs and compute mean/std/sem curves."""
    if not run_frames:
        return pd.DataFrame()
    # Intersect steps across all runs
    step_sets = None
    if debug:
        print(f"[DEBUG] Aggregating value='{value_col}' on step_col='{step_col}' across runs: {list(run_frames.keys())}")
    for df in run_frames.values():
        s = set(df[step_col].astype(int).values.tolist())
        step_sets = s if step_sets is None else (step_sets & s)
    intersect_steps = sorted(list(step_sets)) if step_sets else []
    if align == "intersection":
        common_steps = intersect_steps
        if not common_steps:
            if debug:
                run_names = list(run_frames.keys())
                sizes = {name: len(df[step_col].unique()) for name, df in run_frames.items()}
                print(f"[DEBUG] No common steps across runs {run_names}. Unique counts per run: {sizes}. Consider reducing smoothing or verifying step alignment.")
            return pd.DataFrame()
    else:
        # Union alignment: take all unique steps across runs
        union_steps = sorted({int(x) for df in run_frames.values() for x in df[step_col].astype(int).values.tolist()})
        common_steps = union_steps

    # Build matrix values per step
    matrices = []
    run_order: list[str] = []
    for name, df in run_frames.items():
        if align == "intersection":
            # Ensure unique step entries and align strictly to common_steps order
            sub = df[[step_col, value_col]].dropna().copy()
            sub[step_col] = sub[step_col].astype(int)
            sub = sub.drop_duplicates(subset=[step_col]).sort_values(step_col)
            # Build a dense vector exactly matching common_steps order
            mapping = dict(zip(sub[step_col].tolist(), sub[value_col].astype(float).tolist()))
            v = np.array([mapping[s] for s in common_steps if s in mapping], dtype=float)
            # If for any reason a step is missing (shouldn't happen with intersection), skip this run
            if len(v) != len(common_steps):
                # Skip inconsistent runs to avoid shape mismatches
                if debug:
                    missing = [s for s in common_steps if s not in mapping]
                    print(f"[DEBUG] Skipping run '{name}' due to missing steps: {missing[:10]}{'...' if len(missing)>10 else ''}")
                continue
        else:
            # Interpolate onto the union step grid
            sub = df[[step_col, value_col]].dropna().copy()
            sub[step_col] = sub[step_col].astype(int)
            sub = sub.drop_duplicates(subset=[step_col]).sort_values(step_col)
            interp = np.interp(common_steps, sub[step_col].values.astype(float), sub[value_col].values.astype(float))
            v = interp
        # Optionally smooth
        if rolling_window and rolling_window > 1:
            v = pd.Series(v).rolling(window=int(rolling_window), min_periods=1).mean().values
        matrices.append(v)
        run_order.append(name)
    if not matrices:
        return pd.DataFrame()
    mat = pd.DataFrame(matrices).T  # shape: [steps, runs]
    mean = mat.mean(axis=1)
    # Ensure finite variance values; when only 1 run, std=zeros
    std = (mat.std(axis=1, ddof=1) if mat.shape[1] > 1 else pd.Series([0.0] * len(mean))).fillna(0.0)
    sem = (std / np.sqrt(mat.shape[1]) if mat.shape[1] > 0 else std).fillna(0.0)
    var = std**2
    out = pd.DataFrame({
        'step': common_steps,
        'mean': mean.values,
        'std': std.values,
        'sem': sem.values,
        'var': var.values,
    })
    if debug:
        num_runs = mat.shape[1]
        print(f"[DEBUG] Number of runs used: {num_runs}; number of aligned steps: {len(common_steps)}")
        print(f"[DEBUG] Variance stats -> min: {float(var.min()) if len(var)>0 else 'n/a'}, max: {float(var.max()) if len(var)>0 else 'n/a'}, mean: {float(var.mean()) if len(var)>0 else 'n/a'}")
        # Print sample step values across runs
        if len(common_steps) > 0 and num_runs > 0:
            sample_indices = sorted(set([0, len(common_steps)//2, len(common_steps)-1]))[:3]
            for idx in sample_indices:
                vals = mat.iloc[idx].values.astype(float)
                eq = bool(np.allclose(vals, vals[0])) if len(vals) > 0 else True
                step_val = common_steps[idx]
                per_run = {run_order[i]: float(vals[i]) for i in range(len(vals))}
                print(f"[DEBUG] Step={step_val} values per run: {per_run} | all_equal={eq}")
    return out

def plot_curve(agg: pd.DataFrame, title: str, y_label: str, output_path: Path | None, shade: str = "std") -> None:
    if agg.empty:
        print("Warning: no data to plot")
        return
    plt.figure(figsize=(10, 6))
    # Use seaborn for the mean line
    sns.lineplot(data=agg, x='step', y='mean', color="#1f77b4", linewidth=2, label="mean")
    if shade in ("std", "both") and 'std' in agg.columns:
        plt.fill_between(agg['step'], agg['mean'] - agg['std'], agg['mean'] + agg['std'], color="#1f77b4", alpha=0.15, label="± std")
    if shade in ("sem", "both") and 'sem' in agg.columns:
        plt.fill_between(agg['step'], agg['mean'] - agg['sem'], agg['mean'] + agg['sem'], color="#1f77b4", alpha=0.10, label="± sem")
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()


def _parse_groups_arg(groups_arg: str | None) -> list[tuple[str, str, str | None]]:
    """Parse --groups like:
    - 'PPO:ppo_seed,PPO_RANDOM:ppo_random'
    - With per-group base override: 'PPO_LLM:ppo_llm_seed@/abs/or/rel/path'
    Returns list of (label, pattern, base_override_or_None)
    """
    if not groups_arg:
        return []
    groups: list[tuple[str, str, str | None]] = []
    for part in str(groups_arg).split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            label, patt = part.split(':', 1)
            base_override: str | None = None
            if '@' in patt:
                patt, base_override = patt.split('@', 1)
                base_override = base_override.strip() or None
            groups.append((label.strip(), patt.strip(), base_override))
        else:
            groups.append((part, part, None))
    return groups


def plot_groups(group_to_agg: dict[str, pd.DataFrame], title: str, y_label: str, output_path: Path | None, shade: str = "std") -> None:
    # Use seaborn color palette
    colors = sns.color_palette("tab10")
    plt.figure(figsize=(10, 6))
    for idx, (label, agg) in enumerate(group_to_agg.items()):
        if agg.empty:
            continue
        color = colors[idx % len(colors)]
        # Use seaborn for each group's mean line
        sns.lineplot(data=agg, x='step', y='mean', color=color, linewidth=2, label=label)
        if shade in ("std", "both") and 'std' in agg.columns:
            plt.fill_between(agg['step'], agg['mean'] - agg['std'], agg['mean'] + agg['std'], color=color, alpha=0.15)
        if shade in ("sem", "both") and 'sem' in agg.columns:
            plt.fill_between(agg['step'], agg['mean'] - agg['sem'], agg['mean'] + agg['sem'], color=color, alpha=0.10)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

def main():
    import argparse
    p = argparse.ArgumentParser(description="Aggregate and plot mean/variance curves across runs in a folder")
    p.add_argument("--base_dir", type=str, required=True, help="Folder containing multiple runs, e.g., runs/ppo or runs/reinforce")
    p.add_argument("--pattern", type=str, default=None, help="Substring to filter subfolders, e.g., 'ppo_random' or 'reinforce_expert'")
    p.add_argument("--groups", type=str, default=None, help="Comma-separated label:pattern pairs, optionally with '@base' per group. Example: 'PPO:ppo_seed,PPO_RANDOM:ppo_random,PPO_EXPERT:ppo_expert,PPO_LLM:ppo_llm_seed@runs/old_runs'")
    p.add_argument("--metric", type=str, choices=["return", "loss", "both"], default="return")
    p.add_argument("--shade", type=str, choices=["std", "sem", "both"], default="std", help="Which uncertainty band to shade (default: std)")
    p.add_argument("--rolling_window", type=int, default=100, help="Smoothing window for mean curve (steps)")
    p.add_argument("--debug", action="store_true", help="Print debug info about runs used and variance values")
    p.add_argument("--align", type=str, choices=["intersection", "union"], default="intersection", help="Step alignment: intersection or union (with interpolation)")
    p.add_argument("--out_prefix", type=str, default=None, help="Optional output file prefix; saved inside base_dir")
    args = p.parse_args()

    base = Path(args.base_dir)
    if not base.exists():
        raise SystemExit(f"Base directory not found: {base}")

    groups = _parse_groups_arg(args.groups)
    multi = len(groups) > 0

    # Returns aggregation
    if args.metric in ("return", "both"):
        if multi:
            grouped: dict[str, pd.DataFrame] = {}
            for label, patt, base_override in groups:
                bdir = Path(base_override).expanduser().resolve() if base_override else base
                runs_ret = load_runs(bdir, patt, file_name="train_returns.csv", step_col="step", value_cols=["return"])  # type: ignore[list-item]
                # Fallback for very old runs with 'episode'
                if not runs_ret:
                    runs_ret = load_runs(bdir, patt, file_name="train_returns.csv", step_col="episode", value_cols=["return"])  # type: ignore[list-item]
                    # Rename to unify
                    for k, df in list(runs_ret.items()):
                        df = df.rename(columns={'episode': 'step'})
                        runs_ret[k] = df
                grouped[label] = aggregate_by_step(runs_ret, step_col="step", value_col="return", rolling_window=args.rolling_window, debug=args.debug, align=args.align)
            out = base / f"{args.out_prefix or 'comparison_multi'}_returns.png"
            title = f"Returns mean±sem ({base.name}, rolling={args.rolling_window})"
            plot_groups(grouped, title=title, y_label="Return", output_path=out, shade=args.shade)
        else:
            runs_ret = load_runs(base, args.pattern, file_name="train_returns.csv", step_col="step", value_cols=["return"])  # type: ignore[list-item]
            # Fallback
            if not runs_ret:
                runs_ret = load_runs(base, args.pattern, file_name="train_returns.csv", step_col="episode", value_cols=["return"])  # type: ignore[list-item]
                for k, df in list(runs_ret.items()):
                    df = df.rename(columns={'episode': 'step'})
                    runs_ret[k] = df
            agg_ret = aggregate_by_step(runs_ret, step_col="step", value_col="return", rolling_window=args.rolling_window, debug=args.debug, align=args.align)
            out = base / f"{args.out_prefix or 'comparison'}_returns.png"
            title = f"Returns mean±sem across runs ({base.name}{f' | {args.pattern}' if args.pattern else ''}, rolling={args.rolling_window})"
            plot_curve(agg_ret, title=title, y_label="Return", output_path=out, shade=args.shade)

    # Loss aggregation (policy loss)
    if args.metric in ("loss", "both"):
        if multi:
            grouped_l: dict[str, pd.DataFrame] = {}
            for label, patt, base_override in groups:
                bdir = Path(base_override).expanduser().resolve() if base_override else base
                # Probe steps vs update
                probe = load_runs(bdir, patt, file_name="losses.csv", step_col="steps", value_cols=["policy_loss"])  # type: ignore[list-item]
                if probe:
                    runs_loss = probe
                    step_col_used = "steps"
                else:
                    probe2 = load_runs(bdir, patt, file_name="losses.csv", step_col="update", value_cols=["policy_loss"])  # type: ignore[list-item]
                    runs_loss = probe2
                    step_col_used = "update"
                grouped_l[label] = aggregate_by_step(runs_loss, step_col=step_col_used, value_col="policy_loss", rolling_window=args.rolling_window, debug=args.debug, align=args.align)
            out = base / f"{args.out_prefix or 'comparison_multi'}_loss.png"
            title = f"Policy loss mean±sem ({base.name}, rolling={args.rolling_window})"
            plot_groups(grouped_l, title=title, y_label="Policy loss", output_path=out, shade=args.shade)
        else:
            # Probe to decide which step column to use
            probe = load_runs(base, args.pattern, file_name="losses.csv", step_col="steps", value_cols=["policy_loss"])  # type: ignore[list-item]
            step_col_used = "steps" if any(True for _ in probe) else None
            if step_col_used is None:
                probe2 = load_runs(base, args.pattern, file_name="losses.csv", step_col="update", value_cols=["policy_loss"])  # type: ignore[list-item]
                if any(True for _ in probe2):
                    runs_loss = probe2
                    step_col_used = "update"
                else:
                    runs_loss = {}
            else:
                runs_loss = probe
            agg_loss = aggregate_by_step(runs_loss, step_col=step_col_used or "steps", value_col="policy_loss", rolling_window=args.rolling_window, debug=args.debug, align=args.align)
            out = base / f"{args.out_prefix or 'comparison'}_loss.png"
            title = f"Policy loss mean±sem across runs ({base.name}{f' | {args.pattern}' if args.pattern else ''}, rolling={args.rolling_window})"
            plot_curve(agg_loss, title=title, y_label="Policy loss", output_path=out, shade=args.shade)

if __name__ == "__main__":
    main()
