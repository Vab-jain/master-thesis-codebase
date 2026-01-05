#!/usr/bin/env python3
"""
Main RL Agent Training Module
Trains RL agents using torch-ac framework with and without hints from Oracle Bot or DSPy module.
"""

import argparse
import time
import datetime
import torch_ac
import tensorboardX
import sys
import torch
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
from utils import device
from model import ACModel
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
from utils.hint_wrapper import HintWrapper
from torch_ac.utils.penv import ParallelEnv

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.cm as cm
import numpy as np
try:
    style.use('seaborn-v0_8')  # Try new seaborn style
except OSError:
    try:
        style.use('seaborn')  # Fallback to old seaborn style
    except OSError:
        style.use('default')  # Fallback to default style

class RLAgentTrainer:
    """Main class for training RL agents with torch-ac framework."""
    
    def __init__(self, config_path: str = "../configs/rl_training_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.results_dir = Path(self.config.get('general', {}).get('results_dir', 'RL_Training_Results'))
        self.results_dir.mkdir(exist_ok=True)
        
    def load_config(self) -> Dict[str, Any]:
        """Load training configuration from YAML file."""
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}. Please create the config file before running training.")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_environment_wrappers(self, training_config: Dict[str, Any]) -> List:
        """Setup environment wrappers based on training configuration."""
        wrappers = [FullyObsWrapper]
        
        if training_config.get('use_rgb', False):
            wrappers.append(RGBImgObsWrapper)
        
        # Add hint wrapper if hints are enabled
        # HintWrapper handles all text encoding internally - no need for TextDesciptionWrapper
        if training_config.get('use_hints', False):
            hint_source = training_config.get('hint_source', 'babyai_bot')  # Fixed: use 'babyai_bot' instead of 'oracle'
            hint_type = training_config.get('hint_type', 'subgoal')
            hint_frequency = training_config.get('hint_frequency', 5)
            hint_probability = training_config.get('hint_probability', 1.0)
            hint_stop_percentage = training_config.get('hint_stop_percentage', 
                                                     self.config.get('general', {}).get('default_hint_stop_percentage', 1.0))
            
            # Use utils.HintWrapper for both oracle and dspy hints
            wrappers.append(lambda env: HintWrapper(
                env, 
                hint_type=hint_type,
                hint_frequency=hint_frequency,
                hint_source=hint_source,  # "babyai_bot" or "dspy"
                hint_probability=hint_probability,
                hint_stop_percentage=hint_stop_percentage
            ))
        
        return wrappers
    
    def create_model_name(self, env_name: str, training_config: Dict[str, Any]) -> str:
        """Create descriptive model name."""
        date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        config_name = training_config['name']
        seed = self.config['general']['seed']
        
        return f"{env_name}_{config_name}_seed{seed}_{date}"
    
    def get_training_params(self, env_name: str) -> Dict[str, Any]:
        """Get training parameters for a specific environment."""
        # Check if environment has specific config
        if env_name in self.config.get('environments', {}):
            env_config = self.config['environments'][env_name]
            if 'training_params' in env_config:
                return env_config['training_params']
        else:
            raise ValueError(f"No training parameters found for environment: {env_name}. Please check your configuration.")

    def train_single_agent(self, env_name: str, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a single RL agent with given configuration."""
        # Get environment-specific training parameters
        params = self.get_training_params(env_name)
        
        print(f"\n🚀 Training RL Agent")
        print(f"Environment: {env_name}")
        print(f"Configuration: {training_config['name']}")
        print(f"Use hints: {training_config.get('use_hints', False)}")
        if training_config.get('use_hints', False):
            hint_source = training_config.get('hint_source', 'babyai_bot')  # Fixed: use 'babyai_bot' instead of 'oracle'
            hint_type = training_config.get('hint_type', 'subgoal')
            hint_frequency = training_config.get('hint_frequency', 5)
            hint_probability = training_config.get('hint_probability', 1.0)
            hint_stop_percentage = training_config.get('hint_stop_percentage', 
                                                     self.config.get('general', {}).get('default_hint_stop_percentage', 1.0))
            print(f"Hint source: {hint_source}")
            print(f"Hint type: {hint_type}")
            print(f"Hint frequency: every {hint_frequency} steps")
            print(f"Hint probability: {hint_probability:.1%}")
            if hint_stop_percentage < 1.0:
                hint_stop_frame = int(params.get('frames', 0) * hint_stop_percentage)
                print(f"Hint stop: {hint_stop_percentage:.0%} of training ({hint_stop_frame:,} frames)")
            else:
                print(f"Hint stop: Never (full training)")
        
        print(f"Training frames: {params.get('frames', 'unknown')}")
        print(f"Learning rate: {params.get('lr', 'unknown')}")
        
        # Create model name and directory
        model_name = self.create_model_name(env_name, training_config)
        # Use configured results directory instead of hardcoded 'storage'
        configured_results_dir = self.config.get('general', {}).get('results_dir', 'RL_Training_Results')
        model_dir = os.path.join(configured_results_dir, model_name)
        
        # Setup logging
        txt_logger = utils.get_txt_logger(model_dir)
        csv_file, csv_logger = utils.get_csv_logger(model_dir)
        tb_writer = tensorboardX.SummaryWriter(model_dir)
        
        # Log configuration
        txt_logger.info(f"Training configuration: {training_config}")
        txt_logger.info(f"Environment: {env_name}")
        txt_logger.info(f"Training parameters: {params}")
        
        # Set seed
        seed = self.config['general']['seed']
        utils.seed(seed)
        txt_logger.info(f"Seed: {seed}")
        txt_logger.info(f"Device: {device}")
        
        # Setup environments
        wrappers = self.setup_environment_wrappers(training_config)
        envs = []
        procs = params.get('procs', 1)
        
        for i in range(procs):
            env = utils.make_env(env_name, seed + 10000 * i, wrappers=wrappers)
            envs.append(env)
        
        env = ParallelEnv(envs)
        txt_logger.info("Environments loaded")
        txt_logger.info(f"Wrappers: {[wrapper.__name__ for wrapper in wrappers]}")
        
        # Load training status
        try:
            # Create directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            status = utils.get_status(model_dir)
        except OSError:
            status = {"num_frames": 0, "update": 0}
        txt_logger.info("Training status loaded")
        
        # Load observations preprocessor
        obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
        if "vocab" in status:
            preprocess_obss.vocab.load_vocab(status["vocab"])
        txt_logger.info("Observations preprocessor loaded")
        
        # Load model
        use_memory = training_config.get('use_memory', False)
        use_text = training_config.get('use_text', True)
        use_rgb = training_config.get('use_rgb', False)
        use_hints = training_config.get('use_hints', False)
        
        acmodel = ACModel(obs_space, envs[0].action_space, use_memory, use_text, use_rgb, use_hints)
        if "model_state" in status:
            acmodel.load_state_dict(status["model_state"])
        acmodel.to(device)
        txt_logger.info("Model loaded")
        txt_logger.info(f"{acmodel}")
        txt_logger.info(f"Model features - Memory: {use_memory}, Text: {use_text}, RGB: {use_rgb}, Hints: {use_hints}")
        
        # Load algorithm
        algo_name = params.get('algo', 'ppo')
        
        if algo_name == "a2c":
            algo = torch_ac.A2CAlgo(
                envs, acmodel, device, params['frames_per_proc'], params['discount'],
                params['lr'], params['gae_lambda'], params['entropy_coef'],
                params['value_loss_coef'], params['max_grad_norm'], 1,  # recurrence=1
                0.99, 1e-8, preprocess_obss  # optim_alpha, optim_eps
            )
        elif algo_name == "ppo":
            algo = torch_ac.PPOAlgo(
                envs, acmodel, device, params['frames_per_proc'], params['discount'],
                params['lr'], params['gae_lambda'], params['entropy_coef'],
                params['value_loss_coef'], params['max_grad_norm'], 1,  # recurrence=1
                1e-8, params['clip_eps'], params['epochs'], params['batch_size'], preprocess_obss
            )
        else:
            raise ValueError(f"Incorrect algorithm name: {algo_name}")
        
        if "optimizer_state" in status:
            algo.optimizer.load_state_dict(status["optimizer_state"])
        txt_logger.info("Optimizer loaded")
        
        # Training loop
        num_frames = status["num_frames"]
        update = status["update"]
        start_time = time.time()
        max_frames = params['frames']
        
        training_logs = []
        
        # Rolling window for win rate tracking
        rolling_window_size = self.config.get('general', {}).get('rolling_window_size', 100)  # Default to 100 if not specified
        min_episodes_for_thresholds = self.config.get('general', {}).get('min_episodes_for_thresholds', 50)  # Minimum episodes before tracking
        recent_returns = []
        total_episodes_completed = 0  # Track total episodes for minimum requirement
        
        # Best performance tracking
        best_rolling_win_rate = 0.0
        best_rolling_win_rate_frame = 0
        best_mean_return = float('-inf')
        best_mean_return_frame = 0
        
        # Success rate threshold tracking
        success_thresholds = self.config.get('general', {}).get('success_rate_thresholds', [0.1, 0.25, 0.5, 0.75, 0.9])
        threshold_reached = {threshold: None for threshold in success_thresholds}  # Store frames when reached
        
        # Hint stopping logic
        hint_stop_percentage = training_config.get('hint_stop_percentage', 
                                                 self.config.get('general', {}).get('default_hint_stop_percentage', 1.0))
        hint_stop_frame = int(max_frames * hint_stop_percentage) if hint_stop_percentage < 1.0 else max_frames + 1
        hints_disabled = False
        
        txt_logger.info(f"🎯 Tracking success rate thresholds: {[f'{t:.0%}' for t in success_thresholds]}")
        txt_logger.info(f"📊 Rolling window size: {rolling_window_size} episodes")
        txt_logger.info(f"📊 Minimum episodes before threshold tracking: {min_episodes_for_thresholds}")
        if training_config.get('use_hints', False) and hint_stop_percentage < 1.0:
            txt_logger.info(f"🚫 Hints will be disabled after {hint_stop_frame:,} frames ({hint_stop_percentage:.0%} of training)")
        
        while num_frames < max_frames:
            # Update model parameters
            update_start_time = time.time()
            exps, logs1 = algo.collect_experiences()
            logs2 = algo.update_parameters(exps)
            logs = {**logs1, **logs2}
            update_end_time = time.time()
            
            num_frames += logs["num_frames"]
            update += 1
            
            # Check if we should disable hints
            if (not hints_disabled and training_config.get('use_hints', False) and 
                hint_stop_percentage < 1.0 and num_frames >= hint_stop_frame):
                # Find and disable hints in all environments
                for env_instance in envs:
                    # Navigate through the wrapper stack to find HintWrapper
                    current_env = env_instance
                    while hasattr(current_env, 'env'):
                        if hasattr(current_env, 'disable_hints'):
                            current_env.disable_hints()
                            break
                        current_env = current_env.env
                hints_disabled = True
                txt_logger.info(f"🚫 HINTS DISABLED: Reached {hint_stop_percentage:.0%} of training at {num_frames:,}/{max_frames:,} frames")
                print(f"🚫 HINTS DISABLED: Reached {hint_stop_percentage:.0%} of training at {num_frames:,}/{max_frames:,} frames")
            
            # Update rolling window with episode returns
            episodes_this_update = len(logs["return_per_episode"])
            total_episodes_completed += episodes_this_update
            
            for episode_return in logs["return_per_episode"]:
                recent_returns.append(episode_return)
                if len(recent_returns) > rolling_window_size:
                    recent_returns.pop(0)
            
            # Calculate rolling win rate
            rolling_win_rate = 0.0
            if recent_returns:
                rolling_win_rate = sum(1 for r in recent_returns if r > 0) / len(recent_returns)
            
            # Update best performance tracking
            if rolling_win_rate > best_rolling_win_rate:
                best_rolling_win_rate = rolling_win_rate
                best_rolling_win_rate_frame = num_frames
            
            # Track best mean return
            current_mean_return = utils.synthesize(logs["return_per_episode"])["mean"]
            if current_mean_return > best_mean_return:
                best_mean_return = current_mean_return
                best_mean_return_frame = num_frames
            
            # Check if any new thresholds have been reached (only after minimum episodes)
            threshold_tracking_active = total_episodes_completed >= min_episodes_for_thresholds and len(recent_returns) >= rolling_window_size
            
            if threshold_tracking_active:
                for threshold in success_thresholds:
                    if threshold_reached[threshold] is None and rolling_win_rate >= threshold:
                        threshold_reached[threshold] = num_frames
                        txt_logger.info(f"🎯 SUCCESS THRESHOLD REACHED: {threshold:.0%} at {num_frames:,} frames (after {total_episodes_completed} episodes)!")
                        print(f"🎯 SUCCESS THRESHOLD REACHED: {threshold:.0%} at {num_frames:,} frames (after {total_episodes_completed} episodes)!")
            elif total_episodes_completed < min_episodes_for_thresholds:
                # Log progress towards minimum episodes
                remaining_episodes = min_episodes_for_thresholds - total_episodes_completed
                if update % 10 == 0:  # Log every 10 updates
                    txt_logger.info(f"📊 Collecting episodes for stable threshold tracking: {total_episodes_completed}/{min_episodes_for_thresholds} ({remaining_episodes} more needed)")
            
            # Log training progress
            if update % params['log_interval'] == 0:
                fps = logs["num_frames"] / (update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                return_per_episode = utils.synthesize(logs["return_per_episode"])
                rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
                
                # Store training log
                training_log = {
                    "update": update,
                    "num_frames": num_frames,
                    "fps": fps,
                    "duration": duration,
                    "return_per_episode": return_per_episode,
                    "reshaped_return_per_episode": rreturn_per_episode,
                    "num_frames_per_episode": num_frames_per_episode,
                    "rolling_win_rate": rolling_win_rate,
                    "rolling_episodes_count": len(recent_returns),
                    "total_episodes_completed": total_episodes_completed,
                    "threshold_tracking_active": threshold_tracking_active,
                    "best_rolling_win_rate": best_rolling_win_rate,
                    "best_mean_return": best_mean_return,
                    "entropy": logs["entropy"],
                    "value": logs["value"],
                    "policy_loss": logs["policy_loss"],
                    "value_loss": logs["value_loss"],
                    "grad_norm": logs["grad_norm"],
                    "threshold_reached": {k: v for k, v in threshold_reached.items() if v is not None},
                    "hints_disabled": hints_disabled
                }
                training_logs.append(training_log)
                
                # Print and log
                header = ["update", "frames", "FPS", "duration"]
                data = [update, num_frames, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["rolling_win_rate", "rolling_episodes", "total_episodes", "threshold_active"]
                data += [rolling_win_rate, len(recent_returns), total_episodes_completed, threshold_tracking_active]
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
                
                # Add threshold info to log
                thresholds_reached_count = sum(1 for v in threshold_reached.values() if v is not None)
                threshold_status = "ACTIVE" if threshold_tracking_active else f"WAIT({min_episodes_for_thresholds - total_episodes_completed})"
                
                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | WR {:.1%} ({}/{}) | T {}/{} [{}] | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                    .format(*data, thresholds_reached_count, len(success_thresholds), threshold_status))
                
                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()
                
                if status["num_frames"] == 0:
                    csv_logger.writerow(header)
                csv_logger.writerow(data)
                csv_file.flush()
                
                for field, value in zip(header, data):
                    tb_writer.add_scalar(field, value, num_frames)
            
            # Save model
            if params['save_interval'] > 0 and update % params['save_interval'] == 0:
                # Save status
                status = {
                    "num_frames": num_frames, "update": update,
                    "model_state": acmodel.state_dict(),
                    "optimizer_state": algo.optimizer.state_dict()
                }
                if hasattr(preprocess_obss, "vocab"):
                    status["vocab"] = preprocess_obss.vocab.vocab
                utils.save_status(status, model_dir)
                
                # Show threshold progress
                thresholds_reached_count = sum(1 for v in threshold_reached.values() if v is not None)
                threshold_status = "ACTIVE" if threshold_tracking_active else f"WAITING for {min_episodes_for_thresholds - total_episodes_completed} more episodes"
                txt_logger.info(f"Status saved - Rolling Win Rate: {rolling_win_rate:.1%} over {len(recent_returns)} episodes (Total: {total_episodes_completed}) | Thresholds: {thresholds_reached_count}/{len(success_thresholds)} [{threshold_status}]")
        
        # Final save
        status = {
            "num_frames": num_frames, "update": update,
            "model_state": acmodel.state_dict(),
            "optimizer_state": algo.optimizer.state_dict()
        }
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir)
        
        # Calculate final rolling win rate
        final_rolling_win_rate = 0.0
        if recent_returns:
            final_rolling_win_rate = sum(1 for r in recent_returns if r > 0) / len(recent_returns)
        
        # Save training summary
        training_summary = {
            "model_name": model_name,
            "environment": env_name,
            "training_config": training_config,
            "training_params": params,
            "total_frames": num_frames,
            "total_updates": update,
            "training_time_seconds": time.time() - start_time,
            "final_rolling_win_rate": final_rolling_win_rate,
            "final_rolling_episodes_count": len(recent_returns),
            "total_episodes_completed": total_episodes_completed,
            "rolling_window_size": rolling_window_size,
            "min_episodes_for_thresholds": min_episodes_for_thresholds,
            "threshold_tracking_was_active": total_episodes_completed >= min_episodes_for_thresholds,
            "best_rolling_win_rate": best_rolling_win_rate,
            "best_rolling_win_rate_frame": best_rolling_win_rate_frame,
            "best_mean_return": best_mean_return,
            "best_mean_return_frame": best_mean_return_frame,
            "success_rate_thresholds": threshold_reached,
            "hint_config": {
                "use_hints": training_config.get('use_hints', False),
                "hint_type": training_config.get('hint_type', 'subgoal'),
                "hint_source": training_config.get('hint_source', 'babyai_bot'),  # Fixed: use 'babyai_bot' instead of 'oracle'
                "hint_frequency": training_config.get('hint_frequency', 5),
                "hint_probability": training_config.get('hint_probability', 1.0),
                "hint_stop_percentage": hint_stop_percentage,
                "hint_stop_frame": hint_stop_frame,
                "hints_were_disabled": hints_disabled
            } if training_config.get('use_hints', False) else None,
            "training_logs": training_logs
        }
        
        summary_path = Path(model_dir) / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        # Print final threshold summary
        print(f"\n🎯 SUCCESS RATE THRESHOLD SUMMARY:")
        for threshold in success_thresholds:
            threshold_frames = threshold_reached[threshold]
            if threshold_frames is not None:
                print(f"   {threshold:.0%}: {threshold_frames:,} frames")
            else:
                print(f"   {threshold:.0%}: Not reached")
        
        txt_logger.info(f"Training completed! Final rolling win rate: {final_rolling_win_rate:.1%} over {len(recent_returns)} episodes (Total episodes: {total_episodes_completed})")
        txt_logger.info(f"🏆 Best rolling win rate: {best_rolling_win_rate:.1%} at {best_rolling_win_rate_frame:,} frames")
        txt_logger.info(f"🏆 Best mean return: {best_mean_return:.3f} at {best_mean_return_frame:,} frames")
        if total_episodes_completed >= min_episodes_for_thresholds:
            txt_logger.info(f"Success rate thresholds reached: {threshold_reached}")
        else:
            txt_logger.info(f"⚠️  Threshold tracking was not active - only {total_episodes_completed} episodes completed (needed {min_episodes_for_thresholds})")
        
        self.create_training_plots(training_summary)
        
        return training_summary
    
    def create_training_plots(self, training_summary: Dict[str, Any]) -> None:
        """Create and save training progress plots."""
        model_name = training_summary['model_name']
        # Use the same configured directory as training
        configured_results_dir = self.config.get('general', {}).get('results_dir', 'RL_Training_Results')
        model_dir = Path(configured_results_dir) / model_name
        training_logs = training_summary['training_logs']
        
        if not training_logs:
            print(f"⚠️  No training logs found for {model_name}")
            return
        
        # Extract data for plotting
        updates = [log['update'] for log in training_logs]
        frames = [log['num_frames'] for log in training_logs]
        episodes = [log.get('total_episodes_completed', 0) for log in training_logs]
        mean_returns = [log['return_per_episode']['mean'] for log in training_logs]
        rolling_win_rates = [log.get('rolling_win_rate', 0) for log in training_logs]
        rolling_episodes = [log.get('rolling_episodes_count', 0) for log in training_logs]
        
        # Apply smoothing to mean returns for better visualization
        def smooth_data(data, window_size=5):
            """Apply moving average smoothing to data."""
            if len(data) < window_size:
                return data
            smoothed = []
            for i in range(len(data)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(data), i + window_size // 2 + 1)
                smoothed.append(sum(data[start_idx:end_idx]) / (end_idx - start_idx))
            return smoothed
        
        smoothed_returns = smooth_data(mean_returns, window_size=5)
        
        # Get threshold data
        threshold_reached = training_summary.get('success_rate_thresholds', {})
        # Get the actual thresholds that were tracked (in case config changed)
        threshold_values = list(threshold_reached.keys()) if threshold_reached else [0.1, 0.25, 0.5, 0.75, 0.9]
        
        # Add threshold lines and markers
        # Generate colors dynamically based on number of thresholds
        if len(threshold_values) > 0:
            threshold_colors = cm.rainbow(np.linspace(0, 1, len(threshold_values)))
        else:
            threshold_colors = ['orange', 'yellow', 'lightgreen', 'green', 'darkgreen']
        
        # Create figure with subplots (3 plots instead of 4)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Training Progress: {model_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Mean Return vs Episodes
        ax1.plot(episodes, mean_returns, 'b-', linewidth=1, alpha=0.4, label='Raw Mean Return')
        ax1.plot(episodes, smoothed_returns, 'b-', linewidth=2, alpha=0.8, label='Smoothed Mean Return')
        
        # Add best performance marker
        best_mean_return = training_summary.get('best_mean_return', 0)
        best_mean_return_frame = training_summary.get('best_mean_return_frame', 0)
        if best_mean_return > 0 and best_mean_return_frame > 0:
            # Find corresponding episode for the best frame
            closest_idx = min(range(len(frames)), key=lambda i: abs(frames[i] - best_mean_return_frame)) if frames else 0
            best_episode = episodes[closest_idx] if closest_idx < len(episodes) else 0
            ax1.scatter([best_episode], [best_mean_return], color='red', s=100, zorder=5, 
                       marker='*', label=f'Best: {best_mean_return:.3f}')
        
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Mean Return')
        ax1.set_title('Mean Return During Training')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        ax1.set_ylim(bottom=0)
        
        # Plot 2: Mean Return vs Frames
        ax2.plot(frames, mean_returns, 'g-', linewidth=1, alpha=0.4, label='Raw')
        ax2.plot(frames, smoothed_returns, 'g-', linewidth=2, alpha=0.8, label='Smoothed')
        ax2.set_xlabel('Frames')
        ax2.set_ylabel('Mean Return')
        ax2.set_title('Mean Return vs Training Frames')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        ax2.set_ylim(bottom=0)
        
        # Plot 3: Rolling Win Rate vs Episodes with threshold markers
        ax3.plot(episodes, [wr * 100 for wr in rolling_win_rates], 'r-', linewidth=2, alpha=0.8, label='Win Rate')
        
        # Add best performance marker
        best_rolling_win_rate = training_summary.get('best_rolling_win_rate', 0)
        best_rolling_win_rate_frame = training_summary.get('best_rolling_win_rate_frame', 0)
        if best_rolling_win_rate > 0 and best_rolling_win_rate_frame > 0:
            # Find corresponding episode for the best frame
            closest_idx = min(range(len(frames)), key=lambda i: abs(frames[i] - best_rolling_win_rate_frame)) if frames else 0
            best_episode = episodes[closest_idx] if closest_idx < len(episodes) else 0
            ax3.scatter([best_episode], [best_rolling_win_rate * 100], color='gold', s=100, zorder=5, 
                       marker='*', label=f'Best: {best_rolling_win_rate:.1%}')
        
        # Add threshold lines and markers (reduced annotations for cleaner look)
        threshold_legend_items = []
        for i, threshold in enumerate(threshold_values):
            # Add horizontal threshold line
            line = ax3.axhline(y=threshold * 100, color=threshold_colors[i], linestyle='--', alpha=0.6, linewidth=1)
            threshold_legend_items.append((line, f'{threshold:.0%}'))
            
            # Add vertical marker if threshold was reached
            if threshold in threshold_reached and threshold_reached[threshold] is not None:
                # Find the corresponding episode number for this frame count
                target_frames = threshold_reached[threshold]
                closest_idx = min(range(len(frames)), key=lambda i: abs(frames[i] - target_frames))
                marker_episodes = episodes[closest_idx] if closest_idx < len(episodes) else 0
                
                ax3.axvline(x=marker_episodes, color=threshold_colors[i], linestyle=':', alpha=0.8, linewidth=2)
                ax3.scatter([marker_episodes], [threshold * 100], color=threshold_colors[i], s=80, zorder=5)
        
        ax3.set_xlabel('Episodes')
        ax3.set_ylabel('Rolling Win Rate (%)')
        ax3.set_title('Rolling Win Rate with Success Thresholds')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        ax3.legend(fontsize=8)
        
        # Add training info as text including threshold summary
        threshold_summary = ""
        for threshold in threshold_values:
            threshold_frames = threshold_reached.get(threshold)
            if threshold_frames is not None:
                threshold_summary += f"{threshold:.0%}: {threshold_frames:,}f  "
            else:
                threshold_summary += f"{threshold:.0%}: --  "
        
        total_episodes = training_summary.get('total_episodes_completed', 'unknown')
        min_episodes = training_summary.get('min_episodes_for_thresholds', 50)
        threshold_tracking_status = "✓ Active" if training_summary.get('threshold_tracking_was_active', False) else f"⚠️ Inactive ({total_episodes}/{min_episodes})"
        
        # Best performance info
        best_win_rate = training_summary.get('best_rolling_win_rate', 0)
        best_mean_return = training_summary.get('best_mean_return', 0)
        
        training_info = (
            f"Environment: {training_summary['environment']}\n"
            f"Config: {training_summary['training_config']['name']}\n"
            f"Total Frames: {training_summary['total_frames']:,}\n"
            f"Total Episodes: {total_episodes}\n"
            f"Training Time: {training_summary['training_time_seconds']:.1f}s\n"
            f"Final Win Rate: {training_summary['final_rolling_win_rate']:.1%}\n"
            f"Best Win Rate: {best_win_rate:.1%}\n"
            f"Best Mean Return: {best_mean_return:.3f}\n"
            f"Threshold Tracking: {threshold_tracking_status}\n"
            f"Thresholds: {threshold_summary}"
        )
        
        # Add text box with training info
        fig.text(0.02, 0.02, training_info, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Make room for the info text
        
        # Save the plot
        plot_path = model_dir / "training_progress.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training plot saved: {plot_path}")
        
        # Create a dedicated threshold tracking plot
        plt.figure(figsize=(12, 8))
        
        # Ensure episodes and rolling_win_rates have the same length
        if len(episodes) != len(rolling_win_rates):
            print(f"⚠️  Warning: episodes length ({len(episodes)}) != rolling_win_rates length ({len(rolling_win_rates)})")
            # Use the minimum length to avoid dimension mismatch
            min_length = min(len(episodes), len(rolling_win_rates))
            episodes_plot = episodes[:min_length]
            rolling_win_rates_plot = rolling_win_rates[:min_length]
        else:
            episodes_plot = episodes
            rolling_win_rates_plot = rolling_win_rates
        
        # Plot win rate over episodes
        if len(episodes_plot) > 0 and len(rolling_win_rates_plot) > 0:
            plt.plot(episodes_plot, [wr * 100 for wr in rolling_win_rates_plot], 'b-', linewidth=3, alpha=0.8, label='Rolling Win Rate')
        
            # Add threshold lines and markers
            for i, threshold in enumerate(threshold_values):
                plt.axhline(y=threshold * 100, color=threshold_colors[i], linestyle='--', alpha=0.6, linewidth=2, 
                           label=f'{threshold:.0%} threshold')
                
                if threshold in threshold_reached and threshold_reached[threshold] is not None:
                    # Find the corresponding episode number for this frame count
                    target_frames = threshold_reached[threshold]
                    closest_idx = min(range(len(frames)), key=lambda i: abs(frames[i] - target_frames))
                    target_episodes = episodes[closest_idx] if closest_idx < len(episodes) else 0
                    
                    plt.axvline(x=target_episodes, color=threshold_colors[i], linestyle=':', alpha=0.8, linewidth=2)
                    plt.scatter([target_episodes], [threshold * 100], color=threshold_colors[i], s=150, zorder=5, 
                              edgecolors='black', linewidth=2)
                    plt.annotate(f'{threshold:.0%} reached\nat episode {target_episodes}', 
                               xy=(target_episodes, threshold * 100),
                               xytext=(20, 20), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor=threshold_colors[i], alpha=0.8),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                               fontsize=10, fontweight='bold')
            
            plt.xlabel('Episodes Completed', fontsize=12)
            plt.ylabel('Rolling Win Rate (%)', fontsize=12)
            plt.title(f'Success Rate Threshold Tracking - {model_name}', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.ylim(0, 100)
            if len(episodes_plot) > 0:
                plt.xlim(0, max(episodes_plot) * 1.05)
            
            # Add summary text
            reached_count = sum(1 for v in threshold_reached.values() if v is not None)
            total_episodes = training_summary.get('total_episodes_completed', 'unknown')
            min_episodes = training_summary.get('min_episodes_for_thresholds', 50)
            threshold_active = training_summary.get('threshold_tracking_was_active', False)
            
            if threshold_active:
                summary_text = f"Thresholds Reached: {reached_count}/{len(threshold_values)}\nTotal Episodes: {total_episodes}"
                text_color = 'lightgreen'
            else:
                summary_text = f"⚠️ Threshold tracking inactive\nEpisodes: {total_episodes}/{min_episodes} (need {min_episodes})"
                text_color = 'orange'
            
            plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=text_color, alpha=0.8), verticalalignment='top')
            
            plt.tight_layout()
            threshold_plot_path = model_dir / "threshold_tracking.png"
            plt.savefig(threshold_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"🎯 Threshold tracking plot saved: {threshold_plot_path}")
        else:
            print(f"⚠️  Skipping threshold tracking plot due to insufficient data")
            plt.close()
        
        # Also create a simple mean return plot for quick viewing
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, mean_returns, 'b-', linewidth=1, alpha=0.4, label='Raw Mean Return')
        plt.plot(episodes, smoothed_returns, 'b-', linewidth=2, alpha=0.8, label='Smoothed Mean Return')
        
        # Add best performance marker
        best_mean_return = training_summary.get('best_mean_return', 0)
        best_mean_return_frame = training_summary.get('best_mean_return_frame', 0)
        if best_mean_return > 0 and best_mean_return_frame > 0:
            # Find corresponding episode for the best frame
            closest_idx = min(range(len(frames)), key=lambda i: abs(frames[i] - best_mean_return_frame)) if frames else 0
            best_episode = episodes[closest_idx] if closest_idx < len(episodes) else 0
            plt.scatter([best_episode], [best_mean_return], color='red', s=100, zorder=5, 
                       marker='*', label=f'Best: {best_mean_return:.3f}')
        
        plt.xlabel('Episodes')
        plt.ylabel('Mean Return')
        plt.title(f'Mean Return During Training - {model_name}')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.ylim(bottom=0)
        
        # Add final performance annotation
        if mean_returns:
            final_return = mean_returns[-1]
            final_episodes = episodes[-1] if episodes else 0
            plt.annotate(f'Final: {final_return:.3f}', 
                        xy=(final_episodes, final_return),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        simple_plot_path = model_dir / "mean_return_progress.png"
        plt.savefig(simple_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Simple return plot saved: {simple_plot_path}")

    def create_comparison_plots(self, all_results: Dict[str, Any]) -> None:
        """Create comparison plots across all trained models."""
        if not all_results or 'results' not in all_results:
            print("⚠️  No results to plot")
            return
        
        # Collect all training data
        all_training_data = []
        
        for env_name, env_results in all_results['results'].items():
            for config_name, result in env_results.items():
                if 'training_logs' in result and result['training_logs']:
                    training_logs = result['training_logs']
                    updates = [log['update'] for log in training_logs]
                    episodes = [log.get('total_episodes_completed', 0) for log in training_logs]
                    mean_returns = [log['return_per_episode']['mean'] for log in training_logs]
                    rolling_win_rates = [log.get('rolling_win_rate', 0) for log in training_logs]
                    
                    all_training_data.append({
                        'env_name': env_name,
                        'config_name': config_name,
                        'model_name': result['model_name'],
                        'updates': updates,
                        'episodes': episodes,
                        'mean_returns': mean_returns,
                        'rolling_win_rates': rolling_win_rates,
                        'final_win_rate': result.get('final_rolling_win_rate', 0)
                    })
        
        if not all_training_data:
            print("⚠️  No training data found for comparison plots")
            return
        
        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Training Comparison Across All Models', fontsize=16, fontweight='bold')
        
        # Colors for different configurations
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Plot 1: Mean Returns Comparison
        for i, data in enumerate(all_training_data):
            color = colors[i % len(colors)]
            label = f"{data['env_name']} - {data['config_name']}"
            ax1.plot(data['episodes'], data['mean_returns'], 
                    color=color, linewidth=2, alpha=0.8, label=label)
        
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Mean Return')
        ax1.set_title('Mean Return Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.set_ylim(bottom=0)
        
        # Plot 2: Rolling Win Rate Comparison
        for i, data in enumerate(all_training_data):
            color = colors[i % len(colors)]
            label = f"{data['env_name']} - {data['config_name']}"
            win_rates_percent = [wr * 100 for wr in data['rolling_win_rates']]
            ax2.plot(data['episodes'], win_rates_percent, 
                    color=color, linewidth=2, alpha=0.8, label=label)
        
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Rolling Win Rate (%)')
        ax2.set_title('Rolling Win Rate Comparison')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_plot_path = self.results_dir / "training_comparison.png"
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison plot saved: {comparison_plot_path}")
        
        # Create a summary table plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare summary data
        summary_data = []
        headers = ['Environment', 'Configuration', 'Final Win Rate', 'Final Mean Return', 'Total Updates']
        
        for data in all_training_data:
            final_return = data['mean_returns'][-1] if data['mean_returns'] else 0
            summary_data.append([
                data['env_name'],
                data['config_name'],
                f"{data['final_win_rate']:.1%}",
                f"{final_return:.3f}",
                f"{data['updates'][-1] if data['updates'] else 0}"
            ])
        
        # Create table
        table = ax.table(cellText=summary_data, colLabels=headers, 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Training Results Summary', fontsize=16, fontweight='bold', pad=20)
        
        # Save summary table
        summary_plot_path = self.results_dir / "training_summary_table.png"
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📋 Summary table saved: {summary_plot_path}")

    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all training experiments defined in config."""
        # Get enabled environments
        enabled_envs = []
        environments_config = self.config.get('environments', {})
        
        if isinstance(environments_config, dict):
            # New format: environment-specific configs
            for env_name, env_config in environments_config.items():
                if env_config.get('enabled', False):
                    enabled_envs.append(env_name)
        else:
            # Old format: list of environment names
            enabled_envs = environments_config
        
        print("🎯 Starting RL Agent Training Experiments")
        print(f"Enabled environments: {enabled_envs}")
        print(f"Training configs: {[cfg['name'] for cfg in self.config['training_configs']]}")
        
        all_results = {}
        
        for env_name in enabled_envs:
            print(f"\n🌍 Processing environment: {env_name}")
            
            # Get environment-specific parameters for display
            env_params = self.get_training_params(env_name)
            print(f"   Training frames: {env_params.get('frames', 'unknown')}")
            print(f"   Learning rate: {env_params.get('lr', 'unknown')}")
            print(f"   Processes: {env_params.get('procs', 'unknown')}")
            
            env_results = {}
            
            for training_config in self.config['training_configs']:
                try:
                    result = self.train_single_agent(env_name, training_config)
                    env_results[training_config['name']] = result
                    
                except Exception as e:
                    print(f"❌ Training failed for {env_name} - {training_config['name']}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            all_results[env_name] = env_results
        
        # Save overall results summary
        overall_summary = {
            "config": self.config,
            "results": all_results,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        summary_path = self.results_dir / "overall_training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(overall_summary, f, indent=2)
        
        print(f"\n🎉 All experiments completed!")
        print(f"📁 Results saved in: {self.results_dir}")
        print(f"📊 Overall summary: {summary_path}")
        
        self.create_comparison_plots(overall_summary)
        
        return overall_summary

def main():
    parser = argparse.ArgumentParser(description="RL Agent Training with torch-ac")
    parser.add_argument("--config", type=str, default="../configs/rl_training_config.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--env", type=str, default=None,
                       help="Single environment to train on (overrides config)")
    parser.add_argument("--training-config", type=str, default=None,
                       help="Single training config to use (overrides config)")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = RLAgentTrainer(args.config)
    
    if args.env and args.training_config:
        # Train single agent
        training_configs = {cfg['name']: cfg for cfg in trainer.config['training_configs']}
        if args.training_config not in training_configs:
            print(f"❌ Training config '{args.training_config}' not found")
            print(f"Available configs: {list(training_configs.keys())}")
            return
        
        result = trainer.train_single_agent(args.env, training_configs[args.training_config])
        print(f"✅ Training completed for {args.env} - {args.training_config}")
        
    else:
        # Run all experiments
        results = trainer.run_all_experiments()
        
        # Print summary
        print(f"\n📋 TRAINING SUMMARY")
        print(f"{'='*60}")
        for env_name, env_results in results['results'].items():
            print(f"\n🌍 {env_name}:")
            for config_name, result in env_results.items():
                final_eval = result['final_rolling_win_rate']
                print(f"  {config_name}: {final_eval:.1%} rolling win rate")

if __name__ == "__main__":
    main() 