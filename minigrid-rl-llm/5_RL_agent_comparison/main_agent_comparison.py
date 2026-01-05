#!/usr/bin/env python3
"""
Simplified RL Agent Training Curves Plotter
Generates simple training curves showing mean return and win rate for specified experiments.
"""

import json
import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SimpleTrainingPlotter:
    """Simple class for plotting training curves."""
    
    def __init__(self, config_path: str = "../configs/agent_comparison_config.yaml"):
        self.config = self.load_config(config_path)
        self.training_results_dir = Path(self.config['training_results_dir'])
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_training_data(self, environment: str, config: str) -> dict:
        """Load training data for a specific environment and config."""
        # Try to load from overall summary first
        summary_file = self.training_results_dir / "overall_training_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data = json.load(f)
                results = data.get('results', {})
                if environment in results and config in results[environment]:
                    return results[environment][config]
        
        # Fallback: look for individual model directories
        for model_dir in self.training_results_dir.glob("*"):
            if not model_dir.is_dir():
                continue
                
            summary_file = model_dir / "training_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    if (summary.get('environment') == environment and 
                        summary.get('training_config', {}).get('name') == config):
                        return summary
        
        return None
    
    def plot_training_curves(self, environment: str, configs: list, group_suffix: str = ""):
        """Plot training curves for mean return and win rate for given configs."""
        print(f"📊 Plotting training curves for {environment} - {configs}")
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(
            self.config['plot_settings']['figure_width'], 
            self.config['plot_settings']['figure_height']
        ))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        found_data = False
        
        for i, config in enumerate(configs):
            training_data = self.load_training_data(environment, config)
            
            if training_data is None:
                print(f"  ⚠️  No data found for {environment} - {config}")
                continue
                
            training_logs = training_data.get('training_logs', [])
            if not training_logs:
                print(f"  ⚠️  No training logs found for {environment} - {config}")
                continue
            
            found_data = True
            color = colors[i % len(colors)]
            
            # Extract data
            updates = [log['update'] for log in training_logs]
            mean_returns = [log['return_per_episode']['mean'] for log in training_logs]
            win_rates = [log.get('rolling_win_rate', 0) * 100 for log in training_logs]  # Convert to percentage
            
            label = config.replace('_', ' ').title()
            
            # Plot mean return
            ax1.plot(updates, mean_returns, color=color, linewidth=2, label=label, alpha=0.8)
            
            # Plot win rate
            ax2.plot(updates, win_rates, color=color, linewidth=2, label=label, alpha=0.8)
        
        if not found_data:
            print(f"  ❌ No training data found for {environment}")
            plt.close(fig)
            return
        
        # Configure mean return plot
        ax1.set_xlabel('Training Updates')
        ax1.set_ylabel('Mean Return')
        ax1.set_title(f'{environment} - Mean Return')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        if self.config['plot_settings']['show_legend']:
            ax1.legend()
        
        # Configure win rate plot
        ax2.set_xlabel('Training Updates')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_title(f'{environment} - Win Rate')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        if self.config['plot_settings']['show_legend']:
            ax2.legend()
        
        # Add overall title
        title_suffix = f" ({group_suffix})" if group_suffix else ""
        fig.suptitle(f'{environment} Training Progress{title_suffix}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot with unique filename
        base_filename = f"{environment.replace('-', '_')}_training_curves"
        if group_suffix:
            filename = f"{base_filename}_{group_suffix}.{self.config['plot_settings']['save_format']}"
        else:
            filename = f"{base_filename}.{self.config['plot_settings']['save_format']}"
        
        plot_path = self.output_dir / filename
        plt.savefig(plot_path, dpi=self.config['plot_settings']['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {plot_path}")
    
    def run(self):
        """Run the plotting for all experiments specified in config."""
        print("🚀 Starting Simple Training Curve Plotting")
        print(f"📁 Output directory: {self.output_dir}")
        
        experiments = self.config.get('experiments_to_plot', [])
        
        if not experiments:
            print("❌ No experiments specified in config")
            return
        
        # Track environment occurrence to create unique filenames
        env_count = {}
        
        for experiment in experiments:
            environment = experiment['environment']
            configs = experiment['configs']
            
            # Determine group suffix based on configs
            if any('text' in config for config in configs):
                group_suffix = "with_text"
            else:
                group_suffix = "no_text"
            
            # If we've seen this environment before, we need a suffix
            if environment in env_count:
                env_count[environment] += 1
            else:
                env_count[environment] = 1
            
            try:
                self.plot_training_curves(environment, configs, group_suffix)
            except Exception as e:
                print(f"❌ Error plotting {environment}: {e}")
        
        print("🎉 All plots completed!")

def main():
    """Main function."""
    try:
        plotter = SimpleTrainingPlotter()
        plotter.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 