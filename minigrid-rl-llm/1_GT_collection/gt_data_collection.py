#!/usr/bin/env python3
"""
Ground Truth (GT) Data Collection Script
Collects BabyAI bot demonstrations and creates unified train/test datasets.
Results are saved in GT_dataset/dataset_x/ directory with train.json, test.json, and stats.txt.
"""

import os
import sys
import json
import argparse
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.observation_encoder import ObservationEncoder
from minigrid.utils.baby_ai_bot import BabyAIBot
from minigrid.wrappers import FullyObsWrapper
import gymnasium as gym

def convert_numpy_to_list(obj):
    """Convert numpy arrays and numpy scalars to lists or native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj

class GTDataCollector:
    """Collects ground truth data from BabyAI bot demonstrations."""
    
    def __init__(self, output_dir: str = "GT_dataset"):
        self.output_dir = output_dir
        self.encoder = ObservationEncoder()
        os.makedirs(output_dir, exist_ok=True)
    
    def collect_demonstration(self, env_id: str, seed: int = 42, episode_seed: int = None, 
                             max_steps_per_episode: int = 1000) -> Dict[str, Any]:
        """Collect a single demonstration from BabyAI bot."""
        if episode_seed is not None:
            print(f"Collecting demonstration for {env_id} with seed {seed}, episode {episode_seed}")
        else:
            print(f"Collecting demonstration for {env_id} with seed {seed}")
        
        # Create environment
        env = gym.make(env_id)
        # Wrap with FullyObsWrapper to make agent visible in observations
        env = FullyObsWrapper(env)
        
        # Use episode_seed for environment generation if provided, otherwise use main seed
        env_seed = episode_seed if episode_seed is not None else seed
        env.reset(seed=env_seed)
        
        # Initialize BabyAI bot (using unwrapped env for the bot)
        bot = BabyAIBot(env.unwrapped)
        
        # Collect demonstration
        demonstration = {
            "env_id": env_id,
            "seed": seed,
            "episode_seed": episode_seed,
            "timestamp": datetime.now().isoformat(),
            "episodes": []
        }
        
        episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": []
        }
        
        obs, info = env.reset(seed=env_seed)
        done = False
        truncated = False
        step_count = 0
        
        while not (done or truncated) and step_count < max_steps_per_episode:  # Configurable safety limit
            # Encode observation with all text formats
            encoded_obs = self.encoder.encode_all(obs)
            
            # Get action from BabyAI bot
            action, subgoal = bot.replan()
            
            # Store current observation (before taking step)
            current_obs = convert_numpy_to_list(obs)
            current_info = convert_numpy_to_list(info)
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            
            # Get subgoal for the next state (this is where "done" subgoal would be returned)
            if not (done or truncated):
                next_action, next_subgoal = bot.replan()
                # Use the next subgoal as the subgoal for this step
                subgoal = next_subgoal
            
            # Store step data
            step_data = {
                "step": step_count,
                "observation": current_obs,  # Observation before the step
                "encoded_observations": encoded_obs,
                "action": int(action),
                "subgoal": subgoal,  # Subgoal for the next state
                "reward": reward,
                "done": done,
                "info": current_info  # Info before the step
            }
            
            # Add step data to episode
            episode["observations"].append(step_data)
            
            step_count += 1
        
        demonstration["episodes"].append(episode)
        env.close()
        
        print(f"Collected {step_count} steps")
        return demonstration
    
    def collect_dataset(self, env_ids: List[str], seeds: List[int] = [42], 
                       train_ratio: float = 0.8, episodes_per_seed: int = 1,
                       max_steps_per_episode: int = 1000) -> str:
        """Collect demonstrations and create unified train/test datasets."""
        print(f"Starting GT data collection for {len(env_ids)} environments with {len(seeds)} seeds each")
        print(f"Collecting {episodes_per_seed} episodes per environment-seed combination")
        print(f"Maximum steps per episode: {max_steps_per_episode}")
        
        # Generate all possible environment-seed-episode combinations
        all_combinations = []
        for env_id in env_ids:
            for seed in seeds:
                for episode_idx in range(episodes_per_seed):
                    # Use different episode seeds for variety
                    episode_seed = seed + episode_idx * 1000 if episodes_per_seed > 1 else None
                    all_combinations.append((env_id, seed, episode_seed))
        
        print(f"Total possible combinations: {len(all_combinations)}")
        print(f"Collecting all {len(all_combinations)} combinations")
        
        all_demonstrations = []
        
        for env_id, seed, episode_seed in all_combinations:
            try:
                demonstration = self.collect_demonstration(env_id, seed, episode_seed, max_steps_per_episode)
                all_demonstrations.append(demonstration)
                if episode_seed is not None:
                    print(f"Collected demonstration for {env_id} seed {seed} episode {episode_seed}")
                else:
                    print(f"Collected demonstration for {env_id} seed {seed}")
            except Exception as e:
                print(f"Error collecting demonstration for {env_id} seed {seed}: {e}")
                continue
        
        if not all_demonstrations:
            print("No demonstrations collected!")
            return None
        
        # Create dataset directory with meaningful name
        timestamp = datetime.now().strftime("%m%d_%H%M")  # Shorter timestamp
        num_envs = len(env_ids)
        num_seeds = len(seeds)
        actual_samples = len(all_demonstrations)
        version = "v1"  # Can be made configurable
        
        # Include episodes per seed in dataset name
        dataset_name = f"dataset_{num_envs}env_{num_seeds}seed_{episodes_per_seed}episodes_{version}_{timestamp}"
            
        dataset_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Split into train and test
        random.shuffle(all_demonstrations)
        split_idx = int(len(all_demonstrations) * train_ratio)
        train_demos = all_demonstrations[:split_idx]
        test_demos = all_demonstrations[split_idx:]
        
        # Save train dataset
        train_file = os.path.join(dataset_dir, "train.json")
        with open(train_file, 'w') as f:
            json.dump(train_demos, f, indent=2)
        
        # Save test dataset
        test_file = os.path.join(dataset_dir, "test.json")
        with open(test_file, 'w') as f:
            json.dump(test_demos, f, indent=2)
        
        # Generate statistics
        stats = self.generate_statistics(all_demonstrations, train_demos, test_demos, episodes_per_seed, max_steps_per_episode)
        
        # Save statistics
        stats_file = os.path.join(dataset_dir, "stats.txt")
        with open(stats_file, 'w') as f:
            f.write(stats)
        
        print(f"\n✅ Dataset created successfully!")
        print(f"📁 Dataset directory: {dataset_dir}")
        print(f"📊 Total samples collected: {actual_samples}")
        print(f"📊 Episodes per seed: {episodes_per_seed}")
        print(f"📊 Train samples: {len(train_demos)}")
        print(f"📊 Test samples: {len(test_demos)}")
        print(f"📈 Statistics saved to: {stats_file}")
        
        return dataset_name
    
    def generate_statistics(self, all_demos: List[Dict], train_demos: List[Dict], 
                           test_demos: List[Dict], episodes_per_seed: int = 1,
                           max_steps_per_episode: int = 1000) -> str:
        """Generate comprehensive statistics about the dataset."""
        stats_lines = []
        stats_lines.append("=" * 60)
        stats_lines.append("BABYAI GT DATASET STATISTICS")
        stats_lines.append("=" * 60)
        stats_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        stats_lines.append("")
        
        stats_lines.append("EPISODE CONFIGURATION:")
        stats_lines.append(f"  Episodes per seed: {episodes_per_seed}")
        stats_lines.append(f"  Maximum steps per episode: {max_steps_per_episode}")
        stats_lines.append("")
        
        # Environment statistics
        env_counts = Counter()
        for demo in all_demos:
            env_counts[demo["env_id"]] += 1
        
        stats_lines.append("ENVIRONMENTS:")
        for env, count in env_counts.items():
            stats_lines.append(f"  {env}: {count} demonstrations")
        stats_lines.append("")
        
        # Overall statistics
        stats_lines.append("DATASET SPLITS:")
        stats_lines.append(f"  Total demonstrations: {len(all_demos)}")
        stats_lines.append(f"  Train demonstrations: {len(train_demos)}")
        stats_lines.append(f"  Test demonstrations: {len(test_demos)}")
        stats_lines.append("")
        
        # Subgoal statistics
        subgoal_counts = Counter()
        action_counts = Counter()
        total_steps = 0
        
        for demo in all_demos:
            for episode in demo["episodes"]:
                for step_data in episode["observations"]:
                    subgoal = step_data.get("subgoal", "unknown")
                    action = step_data.get("action", "unknown")
                    subgoal_counts[subgoal] += 1
                    action_counts[action] += 1
                    total_steps += 1
        
        stats_lines.append("SUBGOAL DISTRIBUTION:")
        for subgoal, count in sorted(subgoal_counts.items()):
            percentage = (count / total_steps) * 100
            stats_lines.append(f"  {subgoal}: {count} ({percentage:.1f}%)")
        stats_lines.append("")
        
        stats_lines.append("ACTION DISTRIBUTION:")
        for action, count in sorted(action_counts.items()):
            percentage = (count / total_steps) * 100
            stats_lines.append(f"  {action}: {count} ({percentage:.1f}%)")
        stats_lines.append("")
        
        stats_lines.append(f"TOTAL STEPS: {total_steps}")
        stats_lines.append("")
        
        # Train/Test split statistics
        train_subgoal_counts = Counter()
        train_action_counts = Counter()
        train_steps = 0
        
        for demo in train_demos:
            for episode in demo["episodes"]:
                for step_data in episode["observations"]:
                    subgoal = step_data.get("subgoal", "unknown")
                    action = step_data.get("action", "unknown")
                    train_subgoal_counts[subgoal] += 1
                    train_action_counts[action] += 1
                    train_steps += 1
        
        test_subgoal_counts = Counter()
        test_action_counts = Counter()
        test_steps = 0
        
        for demo in test_demos:
            for episode in demo["episodes"]:
                for step_data in episode["observations"]:
                    subgoal = step_data.get("subgoal", "unknown")
                    action = step_data.get("action", "unknown")
                    test_subgoal_counts[subgoal] += 1
                    test_action_counts[action] += 1
                    test_steps += 1
        
        stats_lines.append("TRAIN SET STATISTICS:")
        stats_lines.append(f"  Total steps: {train_steps}")
        stats_lines.append("  Subgoal distribution:")
        for subgoal, count in sorted(train_subgoal_counts.items()):
            percentage = (count / train_steps) * 100 if train_steps > 0 else 0
            stats_lines.append(f"    {subgoal}: {count} ({percentage:.1f}%)")
        stats_lines.append("")
        
        stats_lines.append("TEST SET STATISTICS:")
        stats_lines.append(f"  Total steps: {test_steps}")
        stats_lines.append("  Subgoal distribution:")
        for subgoal, count in sorted(test_subgoal_counts.items()):
            percentage = (count / test_steps) * 100 if test_steps > 0 else 0
            stats_lines.append(f"    {subgoal}: {count} ({percentage:.1f}%)")
        stats_lines.append("")
        
        stats_lines.append(f"TOTAL STEPS: {total_steps}")
        stats_lines.append("")
        
        stats_lines.append("=" * 60)
        
        return "\n".join(stats_lines)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Collect Ground Truth data from BabyAI bot")
    
    parser.add_argument("--config", type=str, default="configs/gt_collection_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--env-ids", nargs="+", 
                       default=["BabyAI-GoToObj-v0", "BabyAI-GoToLocal-v0", "BabyAI-GoToImpUnlock-v0"],
                       help="List of environment IDs to collect data from")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                       help="List of seeds to use for data collection")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Ratio of data to use for training (default: 0.8)")
    parser.add_argument("--episodes-per-seed", type=int, default=1,
                       help="Number of episodes to collect per environment-seed combination (default: 1)")
    parser.add_argument("--max-steps-per-episode", type=int, default=1000,
                       help="Maximum number of steps per episode (default: 1000)")
    parser.add_argument("--output-dir", type=str, default="GT_dataset",
                       help="Directory to save collected data")
    
    args = parser.parse_args()
    
    # Load configuration or use command line arguments
    if os.path.exists(args.config):
        config = load_config(args.config)
        env_ids = config.get("environments", args.env_ids)
        seeds = config.get("seeds", args.seeds)
        train_ratio = config.get("train_ratio", args.train_ratio)
        episodes_per_seed = config.get("episodes_per_seed", args.episodes_per_seed)
        max_steps_per_episode = config.get("max_steps_per_episode", args.max_steps_per_episode)
    else:
        env_ids = args.env_ids
        seeds = args.seeds
        train_ratio = args.train_ratio
        episodes_per_seed = args.episodes_per_seed
        max_steps_per_episode = args.max_steps_per_episode
    
    # Initialize collector
    collector = GTDataCollector(args.output_dir)
    
    # Collect dataset
    dataset_name = collector.collect_dataset(env_ids, seeds, train_ratio, episodes_per_seed, max_steps_per_episode)
    
    if dataset_name:
        print(f"\n✅ GT data collection completed!")
        print(f"📁 Dataset saved as: {dataset_name}")
        print(f"📁 Full path: {os.path.join(args.output_dir, dataset_name)}")
    else:
        print("❌ GT data collection failed!")

if __name__ == "__main__":
    main() 