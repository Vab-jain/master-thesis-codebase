#!/usr/bin/env python3
"""
Paper Figure Generator for LLM-guided RL Training Results
========================================================
Generates the specific figures required for the LaTeX document based on experimental results.

Figures produced:
1. Training curves (2×3 grid): Win-rate progression for f=5
2. Training curves appendix (2×3 grid): Win-rate progression for f=10  
3. Learning efficiency (2×3 grid): Frames-to-threshold bar charts
4. Frequency sweeps (1×3 grid): Oracle vs LLM frequency comparison

Usage:
    python plot_comparisons_paper.py \
        --gotoobj_dir 4_RL_agent_training/RL_Training_Results_ALL_final_GoToObj \
        --opendoor_dir 4_RL_agent_training/RL_Training_Results_ALL_final_OpenDoor \
        --pickuploc_dir 4_RL_agent_training/RL_Training_Results_ALL_final_PickupLoc \
        --output_dir 5_RL_agent_comparison/paper_figures
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configure matplotlib and seaborn for paper-quality figures
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'lines.linewidth': 2,
    'lines.markersize': 6
})
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
COLORS = sns.color_palette("deep", 10)

def parse_summary(path: Path) -> Optional[dict]:
    """Load a training_summary.json file and return its data dict."""
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None

def collect_runs_by_env(env_dir: Path, env_name: str) -> Dict[str, dict]:
    """Collect all runs for a specific environment."""
    runs = {}
    
    for summary_path in env_dir.rglob("training_summary.json"):
        summary = parse_summary(summary_path)
        if summary is None:
            continue
            
        cfg = summary.get("training_config", {})
        use_hints = cfg.get("use_hints", False)
        use_text = cfg.get("use_text", False)
        hint_freq = cfg.get("hint_frequency", 0) or 0
        hint_prob = float(cfg.get("hint_probability", 0)) if use_hints else 0.0
        
        # Determine configuration type based on hint probability
        if not use_hints:
            config_type = "baseline"
        elif hint_prob == 1.0:
            config_type = "oracle"
        elif abs(hint_prob - 0.5) < 1e-3:
            config_type = "llm"  # Main LLM configuration (50% probability)
        elif abs(hint_prob - 0.2) < 1e-3:
            config_type = "llm02"  # Alternative LLM configuration (20% probability) 
        else:
            print(f"⚠️  Skipping unknown hint probability: {hint_prob}")
            continue  # Skip other configurations
            
        # Create unique key
        text_suffix = "_text" if use_text else ""
        key = f"{config_type}_f{hint_freq}{text_suffix}"
        
        runs[key] = {
            "config_type": config_type,
            "hint_freq": hint_freq,
            "use_text": use_text,
            "hint_prob": hint_prob,
            "summary": summary,
            "env": env_name
        }
    
    return runs

def extract_training_curve(summary: dict) -> Tuple[List[int], List[float]]:
    """Extract training curve data (num_frames, win_rate) - using num_frames for complete x-axis."""
    logs = summary.get("training_logs", [])
    if not logs:
        return [], []
    
    # Use num_frames directly from training logs (this should give complete range)
    num_frames = []
    win_rates = []
    
    for entry in logs:
        if "num_frames" in entry and "rolling_win_rate" in entry:
            num_frames.append(entry["num_frames"])
            win_rates.append(entry["rolling_win_rate"] * 100)  # Convert to percentage
    
    return num_frames, win_rates

def extract_thresholds(summary: dict) -> Dict[str, Optional[int]]:
    """Extract frames-to-threshold data."""
    return summary.get("success_rate_thresholds", {})

def plot_training_curves(all_runs: Dict[str, Dict[str, dict]], freq: int, output_path: Path, title_suffix: str = ""):
    """Generate 2×3 grid of training curves for specific frequency."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    environments = ["GoToObj", "OpenDoor", "PickupLoc"]
    
    for col, env in enumerate(environments):
        env_runs = all_runs[env]
        
        for row, use_text in enumerate([False, True]):
            ax = axes[row, col]
            text_suffix = "_text" if use_text else ""
            
            # Get runs for this configuration - use main LLM config (50% probability)
            baseline_key = f"baseline_f0{text_suffix}"
            oracle_key = f"oracle_f{freq}{text_suffix}"
            llm_key = f"llm_f{freq}{text_suffix}"
            
            # Plot curves
            for key, label, color in [
                (baseline_key, "Baseline", COLORS[0]),
                (oracle_key, f"Oracle f={freq}", COLORS[1]),
                (llm_key, f"LLM-hints f={freq}", COLORS[2])
            ]:
                if key in env_runs:
                    num_frames, win_rates = extract_training_curve(env_runs[key]["summary"])
                    if num_frames:
                        # Convert to millions for readability
                        frames_millions = [x / 1e6 for x in num_frames]
                        ax.plot(frames_millions, win_rates, label=label, color=color, linewidth=2)
                        print(f"✅ Plotted {env} {key}: {len(num_frames)} points, max frames: {max(num_frames)/1e6:.1f}M")
                    else:
                        print(f"⚠️  No data for {env} {key}")
                else:
                    print(f"⚠️  Missing {env} {key}")
            
            # Formatting
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            
            if row == 0:
                ax.set_title(f"{env}", fontsize=18, fontweight='bold')
            if col == 0:
                text_label = "With Text" if use_text else "No Text"
                ax.set_ylabel(f"{text_label}\nWin Rate (%)", fontsize=16, fontweight='bold')
            if row == 1:
                ax.set_xlabel("Training Frames (M)", fontsize=16)
            if row == 0 and col == 2:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    
    plt.tight_layout()
    fig.suptitle(f"Training Curves: Win-Rate Progression {title_suffix}", fontsize=20, y=0.98)
    plt.subplots_adjust(top=0.93)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {output_path}")

def plot_learning_efficiency_f5(all_runs: Dict[str, Dict[str, dict]], output_path: Path):
    """Generate 2×3 grid of frames-to-threshold bar charts for f=5 only (main paper)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    environments = ["GoToObj", "OpenDoor", "PickupLoc"]
    thresholds = [0.25, 0.5, 0.75, 0.9]
    
    # Configuration mapping for f=5 only
    configs = [
        ("baseline_f0", "Baseline", COLORS[0]),
        ("oracle_f1", "Oracle f=1", COLORS[1]),
        ("oracle_f5", "Oracle f=5", COLORS[2]),
        ("llm_f5", "LLM-hints f=5", COLORS[3])
    ]
    
    for col, env in enumerate(environments):
        env_runs = all_runs[env]
        
        for row, use_text in enumerate([False, True]):
            ax = axes[row, col]
            text_suffix = "_text" if use_text else ""
            
            bar_width = 0.18
            x = np.arange(len(thresholds))
            
            for i, (base_key, label, color) in enumerate(configs):
                key = f"{base_key}{text_suffix}"
                
                if key in env_runs:
                    threshold_data = extract_thresholds(env_runs[key]["summary"])
                    y_values = []
                    
                    for thresh in thresholds:
                        frames = threshold_data.get(str(thresh), None)
                        if frames is not None:
                            y_values.append(frames / 1000)  # Convert to K frames
                        else:
                            y_values.append(np.nan)
                    
                    x_pos = x + i * bar_width
                    bars = ax.bar(x_pos, y_values, bar_width, label=label, color=color, 
                                 edgecolor='black', linewidth=0.5, alpha=0.8)
                    
                    # Add value labels on bars
                    for bar, val in zip(bars, y_values):
                        if not np.isnan(val):
                            height = bar.get_height()
                            if height > 0:
                                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                                       f'{val:.0f}K' if val < 1000 else f'{val/1000:.1f}M',
                                       ha='center', va='bottom', fontsize=12, rotation=90)
            
            # Formatting
            ax.set_yscale('log')
            ax.set_xticks(x + bar_width * 1.5)
            ax.set_xticklabels([f"{int(t*100)}%" for t in thresholds])
            ax.grid(True, axis='y', alpha=0.3)
            
            if row == 0:
                ax.set_title(f"{env}", fontsize=18, fontweight='bold')
            if col == 0:
                text_label = "With Text" if use_text else "No Text"
                ax.set_ylabel(f"{text_label}\nFrames (K)", fontsize=16, fontweight='bold')
            if row == 1:
                ax.set_xlabel("Success Threshold", fontsize=16)
            if row == 0 and col == 2:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    
    plt.tight_layout()
    fig.suptitle("Learning Efficiency: Frames to Reach Success Thresholds (f=5)", fontsize=20, y=0.98)
    plt.subplots_adjust(top=0.93)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {output_path}")

def plot_learning_efficiency_full(all_runs: Dict[str, Dict[str, dict]], output_path: Path):
    """Generate 2×3 grid of frames-to-threshold bar charts with all frequencies (appendix)."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    environments = ["GoToObj", "OpenDoor", "PickupLoc"]
    thresholds = [0.25, 0.5, 0.75, 0.9]
    
    # Configuration mapping for all frequencies
    configs = [
        ("baseline_f0", "Baseline", COLORS[0]),
        ("oracle_f1", "Oracle f=1", COLORS[1]),
        ("oracle_f5", "Oracle f=5", COLORS[2]),
        ("llm_f5", "LLM-hints f=5", COLORS[3]),
        ("llm_f10", "LLM-hints f=10", COLORS[4])
    ]
    
    for col, env in enumerate(environments):
        env_runs = all_runs[env]
        
        for row, use_text in enumerate([False, True]):
            ax = axes[row, col]
            text_suffix = "_text" if use_text else ""
            
            bar_width = 0.15
            x = np.arange(len(thresholds))
            
            for i, (base_key, label, color) in enumerate(configs):
                key = f"{base_key}{text_suffix}"
                
                if key in env_runs:
                    threshold_data = extract_thresholds(env_runs[key]["summary"])
                    y_values = []
                    
                    for thresh in thresholds:
                        frames = threshold_data.get(str(thresh), None)
                        if frames is not None:
                            y_values.append(frames / 1000)  # Convert to K frames
                        else:
                            y_values.append(np.nan)
                    
                    x_pos = x + i * bar_width
                    bars = ax.bar(x_pos, y_values, bar_width, label=label, color=color, 
                                 edgecolor='black', linewidth=0.5, alpha=0.8)
                    
                    # Add value labels on bars
                    for bar, val in zip(bars, y_values):
                        if not np.isnan(val):
                            height = bar.get_height()
                            if height > 0:
                                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                                       f'{val:.0f}K' if val < 1000 else f'{val/1000:.1f}M',
                                       ha='center', va='bottom', fontsize=12, rotation=90)
            
            # Formatting
            ax.set_yscale('log')
            ax.set_xticks(x + bar_width * 2)
            ax.set_xticklabels([f"{int(t*100)}%" for t in thresholds])
            ax.grid(True, axis='y', alpha=0.3)
            
            if row == 0:
                ax.set_title(f"{env}", fontsize=18, fontweight='bold')
            if col == 0:
                text_label = "With Text" if use_text else "No Text"
                ax.set_ylabel(f"{text_label}\nFrames (K)", fontsize=16, fontweight='bold')
            if row == 1:
                ax.set_xlabel("Success Threshold", fontsize=16)
            if row == 0 and col == 2:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    
    plt.tight_layout()
    fig.suptitle("Learning Efficiency: Frames to Reach Success Thresholds (All Frequencies)", fontsize=20, y=0.98)
    plt.subplots_adjust(top=0.93)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {output_path}")

def plot_frequency_comparison_curves(all_runs: Dict[str, Dict[str, dict]], output_path: Path):
    """Generate 1×3 grid showing training curves comparing different frequencies."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    environments = ["GoToObj", "OpenDoor", "PickupLoc"]
    
    # Define configurations to compare (no text versions for clarity)
    configs = [
        ("baseline_f0", "Baseline", COLORS[0], '-'),
        ("oracle_f1", "Oracle f=1", COLORS[1], '-'),
        ("oracle_f5", "Oracle f=5", COLORS[2], '-'),
        ("llm_f5", "LLM-hints f=5", COLORS[3], '-'),
        ("llm_f10", "LLM-hints f=10", COLORS[4], '--')
    ]
    
    for col, env in enumerate(environments):
        ax = axes[col]
        env_runs = all_runs[env]
        
        # Plot training curves for each configuration
        for config_key, label, color, linestyle in configs:
            if config_key in env_runs:
                num_frames, win_rates = extract_training_curve(env_runs[config_key]["summary"])
                if num_frames:
                    # Convert to millions for readability
                    frames_millions = [x / 1e6 for x in num_frames]
                    ax.plot(frames_millions, win_rates, label=label, color=color, 
                           linestyle=linestyle, linewidth=2, alpha=0.8)
        
        # Formatting
        ax.set_title(f"{env}", fontsize=22, fontweight='bold')
        ax.set_xlabel("Training Frames (M)", fontsize=20, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        if col == 0:
            ax.set_ylabel("Win Rate (%)", fontsize=20, fontweight='bold')
        if col == 2:
            ax.legend(fontsize=18, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    # fig.suptitle("LLM-Hints Frequency Impact: Higher Frequency Improves Sample Efficiency", fontsize=20, y=1.02)
    # plt.subplots_adjust(top=0.88)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures for LLM-guided RL training results")
    parser.add_argument("--gotoobj_dir", required=True, type=Path, help="GoToObj results directory")
    parser.add_argument("--opendoor_dir", required=True, type=Path, help="OpenDoor results directory")
    parser.add_argument("--pickuploc_dir", required=True, type=Path, help="PickupLoc results directory")
    parser.add_argument("--output_dir", required=True, type=Path, help="Output directory for figures")
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect runs from all environments
    print("📊 Collecting experimental data...")
    all_runs = {
        "GoToObj": collect_runs_by_env(args.gotoobj_dir, "GoToObj"),
        "OpenDoor": collect_runs_by_env(args.opendoor_dir, "OpenDoor"),
        "PickupLoc": collect_runs_by_env(args.pickuploc_dir, "PickupLoc")
    }
    
    # Print available configurations for debugging
    for env, runs in all_runs.items():
        print(f"\n{env} configurations found: {list(runs.keys())}")
        for key, run in runs.items():
            total_frames = run["summary"].get("total_frames", 0)
            print(f"  {key}: {total_frames/1e6:.1f}M frames")
    
    # Generate figures
    print("\n🎨 Generating paper figures...")
    
    # Figure 1: Training curves for f=5 (main paper)
    plot_training_curves(all_runs, 5, args.output_dir / "fig_training_curves_f5.png", "(f=5)")
    
    # Figure 2: Learning efficiency comparison f=5 only (main paper)
    plot_learning_efficiency_f5(all_runs, args.output_dir / "fig_learning_efficiency_f5.png")
    
    # Figure 3: Frequency comparison curves (main paper)
    plot_frequency_comparison_curves(all_runs, args.output_dir / "fig_frequency_comparison.png")
    
    # Appendix figures
    print("\n📚 Generating appendix figures...")
    
    # Appendix A: Training curves for f=10
    plot_training_curves(all_runs, 10, args.output_dir / "fig_training_curves_f10.png", "(f=10 - Appendix)")
    
    # Appendix B: Learning efficiency with all frequencies
    plot_learning_efficiency_full(all_runs, args.output_dir / "fig_learning_efficiency_full.png")
    
    print("\n✅ All paper figures generated successfully!")
    print(f"📁 Output directory: {args.output_dir}")

if __name__ == "__main__":
    main() 