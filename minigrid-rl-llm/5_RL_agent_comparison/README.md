# Simple Training Curve Plotter

This simplified module generates training curves showing mean return and win rate for RL agent experiments.

## Quick Start

1. Update the config file `../configs/agent_comparison_config.yaml` to specify which experiments to plot
2. Run the script: `python main_agent_comparison.py`
3. Find the plots in the `Simple_Training_Plots/` directory

## Configuration

Edit `configs/agent_comparison_config.yaml` to specify:

- `training_results_dir`: Path to your training results
- `output_dir`: Where to save the plots
- `experiments_to_plot`: List of environment and config combinations to plot

Example:
```yaml
experiments_to_plot:
  - environment: 'BabyAI-GoToObj-v0'
    configs: ['baseline', 'oracle_hints_action']
```

## Output

For each experiment, you get a single PNG file with two side-by-side plots:
- Left: Mean Return vs Training Updates
- Right: Win Rate (%) vs Training Updates

Each config gets its own colored line on both plots.

## What Changed

This simplified version removes:
- Complex experiment structures
- Multiple plot types
- Statistical summaries
- CSV exports
- Threshold analysis
- Sample efficiency analysis

It focuses only on the essential training progress visualization. 