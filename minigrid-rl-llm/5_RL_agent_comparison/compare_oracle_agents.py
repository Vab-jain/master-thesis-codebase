#!/usr/bin/env python3
"""
Oracle Agent Comparison Script
Compares baseline, oracle_action, and oracle_subgoal agents without text versions.
Plots training curves showing mean return and rolling win rate side-by-side.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Configure matplotlib for consistent styling with other figures
plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 26,
    'axes.labelsize': 24,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'figure.titlesize': 28,
    'lines.linewidth': 2,
    'lines.markersize': 6
})
sns.set_theme(style="whitegrid", context="paper", font_scale=1.8)

# Custom colors avoiding green
COLORS = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

class OracleAgentComparator:
    """Compare oracle agent types with baseline (no text versions)."""
    
    def __init__(self):
        # Specific directory mappings for GoToObj without text
        self.agent_dirs = {
            "Baseline": Path("../4_RL_agent_training/RL_Training_Results_ALL_final_GoToObj/BabyAI-GoToObj-v0_baseline_seed42_25-07-07-16-09-54"),
            "Oracle Action": Path("../4_RL_agent_training/RL_Training_Results_ALL_final_GoToObj/BabyAI-GoToObj-v0_oracle_hints_action_1_seed42_25-07-07-16-51-19"),
            "Oracle Subgoal": Path("../4_RL_agent_training/RL_Training_Results_oracle_final/BabyAI-GoToObj-v0_oracle_hints_subgoal_seed42_25-06-21-18-40-55")
        }
        
    def load_training_data(self, agent_name: str) -> dict:
        """Load training data for a specific agent."""
        agent_dir = self.agent_dirs[agent_name]
        print(f"  📁 Loading from: {agent_dir.name}")
        
        if not agent_dir.exists():
            print(f"  ❌ Directory does not exist: {agent_dir}")
            return None
            
        summary_file = agent_dir / "training_summary.json"
        if not summary_file.exists():
            print(f"  ❌ No training_summary.json in {agent_dir}")
            return None
            
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
                print(f"  ✅ Loaded training data: {len(data.get('training_logs', []))} log entries")
                return data
        except Exception as e:
            print(f"  ❌ Error loading {summary_file}: {e}")
            return None
    
    def extract_training_curves(self, training_data: dict) -> tuple:
        """Extract training curves from training data."""
        if not training_data or 'training_logs' not in training_data:
            print(f"  ❌ No training_logs in data")
            return [], [], []
            
        logs = training_data['training_logs']
        print(f"  📊 Processing {len(logs)} training log entries...")
        
        # Extract data using num_frames for x-axis (millions), mean return, and rolling win rate (%)
        num_frames = []
        mean_returns = []
        win_rates = []
        
        for i, log in enumerate(logs):
            if 'num_frames' in log and 'return_per_episode' in log:
                num_frames.append(log['num_frames'] / 1e6)  # Convert to millions
                mean_returns.append(log['return_per_episode']['mean'])
                
                # Check for rolling_win_rate, might be missing in some logs
                if 'rolling_win_rate' in log:
                    win_rates.append(log['rolling_win_rate'] * 100)  # Convert to percentage
                else:
                    # Use 0 or skip this entry
                    win_rates.append(0.0)
            else:
                print(f"  ⚠️  Entry {i} missing required fields")
        
        print(f"  ✅ Extracted {len(num_frames)} data points")
        if num_frames:
            print(f"  📈 Frame range: {min(num_frames):.2f}M to {max(num_frames):.2f}M")
            print(f"  📈 Return range: {min(mean_returns):.3f} to {max(mean_returns):.3f}")
            print(f"  📈 Win rate range: {min(win_rates):.1f}% to {max(win_rates):.1f}%")
        
        return num_frames, mean_returns, win_rates
    
    def plot_comparison(self, output_path: str = "paper_figures/oracle_comparison_no_text.png"):
        """Plot comparison of baseline vs oracle agents for GoToObj without text."""
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        plot_data = {}
        
        # Process each agent
        for i, (agent_name, agent_dir) in enumerate(self.agent_dirs.items()):
            print(f"\n📊 Processing {agent_name}...")
            training_data = self.load_training_data(agent_name)
            
            if training_data is None:
                print(f"⚠️  No data found for {agent_name}")
                continue
            
            num_frames, mean_returns, win_rates = self.extract_training_curves(training_data)
            
            if not num_frames:
                print(f"⚠️  No training curve data for {agent_name}")
                continue
            
            color = COLORS[i]
            plot_data[agent_name] = (num_frames, mean_returns, win_rates, color)
            
            print(f"✅ Prepared {agent_name}: {len(num_frames)} points, max frames: {max(num_frames):.1f}M")
        
        if not plot_data:
            print(f"❌ No training data found")
            plt.close(fig)
            return
        
        # Now plot all the data
        print(f"\n🎨 Plotting {len(plot_data)} agents...")
        for agent_name, (num_frames, mean_returns, win_rates, color) in plot_data.items():
            # Plot mean return
            ax1.plot(num_frames, mean_returns, color=color, linewidth=2, label=agent_name, alpha=0.8)
            print(f"  📈 Plotted {agent_name} mean return: {len(num_frames)} points")
            
            # Plot win rate
            ax2.plot(num_frames, win_rates, color=color, linewidth=2, label=agent_name, alpha=0.8)
            print(f"  📈 Plotted {agent_name} win rate: {len(win_rates)} points")
        
        # Configure mean return plot (no title as requested)
        ax1.set_xlabel("Training Frames (M)", fontsize=24, fontweight='bold')
        ax1.set_ylabel("Mean Return", fontsize=24, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        ax1.legend(fontsize=22)
        
        # Configure win rate plot (no title as requested)
        ax2.set_xlabel("Training Frames (M)", fontsize=24, fontweight='bold')
        ax2.set_ylabel("Win Rate (%)", fontsize=24, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=22)
        
        plt.tight_layout()
        
        # Save plot
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_file}")

def main():
    """Main function."""
    print("🚀 Starting Oracle Agent Comparison (GoToObj, No Text)")
    
    comparator = OracleAgentComparator()
    comparator.plot_comparison()
    
    print("\n🎉 Oracle agent comparison completed!")

if __name__ == "__main__":
    main() 