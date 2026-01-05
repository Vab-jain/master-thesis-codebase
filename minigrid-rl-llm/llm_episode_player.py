#!/usr/bin/env python3
"""
Script to test LLM agent playing episodes across multiple environments.
Uses DSPy with GROQ API to control the agent at every step.
Tracks success rates and handles rate limiting.
"""

import time
import json
import dspy
from utils.observation_encoder import ObservationEncoder
from utils.env import make_env
from utils.dspy_signature import SubgoalPredictor, configure_llm
from utils.config_task_desc import task_desc
import utils
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
import numpy as np
from datetime import datetime
import os

class LLMEpisodePlayer:
    def __init__(self, sleep_duration=2.0, max_steps_per_episode=50):
        """
        Initialize the LLM Episode Player.
        
        Args:
            sleep_duration: Time to sleep after each LLM call to avoid rate limits
            max_steps_per_episode: Maximum steps per episode before timeout
        """
        self.sleep_duration = sleep_duration
        self.max_steps_per_episode = max_steps_per_episode
        
        # Configure LLM
        configure_llm()
        self.llm_agent = SubgoalPredictor()
        
        # Initialize observation encoder
        self.encoder = ObservationEncoder()
        
        # Results tracking
        self.results = {
            'episodes': [],
            'summary': {},
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"🤖 LLM Episode Player initialized with {sleep_duration}s sleep between calls")
    
    def play_episode(self, env_key, episode_id=0, verbose=True):
        """
        Play a single episode using LLM agent.
        
        Args:
            env_key: Environment key (e.g., 'BabyAI-GoToObj-v0')
            episode_id: Episode identifier for logging
            verbose: Whether to print step-by-step information
        
        Returns:
            dict: Episode results including success, steps, actions, etc.
        """
        # Create environment
        env = make_env(env_key=env_key)
        env = FullyObsWrapper(env)
        
        # Reset environment
        obs, _ = env.reset()
        all_encodings = self.encoder.encode_all(obs)
        current_state = all_encodings['ascii']
        
        # Episode tracking
        episode_data = {
            'env_key': env_key,
            'episode_id': episode_id,
            'mission': obs.get('mission', ''),
            'steps': 0,
            'actions': [],
            'success': False,
            'termination_reason': 'max_steps',
            'llm_responses': []
        }
        
        previous_actions = []
        step_count = 0
        done = False
        
        if verbose:
            print(f"\n🎮 Episode {episode_id} - {env_key}")
            print(f"Mission: {episode_data['mission']}")
            print("=" * 50)
        
        # Task description for LLM
        task_description = task_desc if task_desc else "You are an agent in a grid environment with access to actions like (0:Turn left, 1: Turn right, 2: move forward, 3: pickup, 4: drop, 5: toggle, 6:done). Following is the current state of the environment:"
        
        while not done and step_count < self.max_steps_per_episode:
            if verbose:
                print(f"\nStep {step_count}:")
                print(current_state)
                print(f"Previous actions: {previous_actions[-3:]}")  # Show last 3 actions
            
            try:
                # Get LLM response
                response = self.llm_agent(
                    task_description=task_description,
                    current_state=current_state,
                    previous_actions=previous_actions[-5:]  # Last 5 actions
                )
                
                # Add sleep to avoid rate limits
                time.sleep(self.sleep_duration)
                
                action = response.primitive_action
                
                if verbose:
                    print(f"LLM Response: {response.reasoning}")
                    print(f"Action: {action}")
                
                # Store LLM response data
                episode_data['llm_responses'].append({
                    'step': step_count,
                    'reasoning': getattr(response, 'reasoning', ''),
                    'action': action,
                    'state': current_state[:200] + "..." if len(current_state) > 200 else current_state
                })
                
                # Take action
                obs, reward, terminated, truncated, _ = env.step(action)
                
                # Update tracking
                step_count += 1
                previous_actions.append(f'Step-{step_count}: {action}')
                episode_data['actions'].append(action)
                episode_data['steps'] = step_count
                
                # Check termination
                if terminated:
                    episode_data['success'] = True
                    episode_data['termination_reason'] = 'success'
                    done = True
                    if verbose:
                        print(f"✅ SUCCESS! Episode completed in {step_count} steps")
                elif truncated:
                    episode_data['termination_reason'] = 'truncated'
                    done = True
                    if verbose:
                        print(f"⚠️  Episode truncated after {step_count} steps")
                
                # Update state for next iteration
                if not done:
                    all_encodings = self.encoder.encode_all(obs)
                    current_state = all_encodings['ascii']
                
            except Exception as e:
                print(f"❌ Error during episode: {e}")
                episode_data['termination_reason'] = f'error: {str(e)}'
                done = True
                # Add extra sleep on error in case it's rate limiting
                time.sleep(self.sleep_duration * 2)
        
        if step_count >= self.max_steps_per_episode:
            if verbose:
                print(f"⏰ Episode timed out after {self.max_steps_per_episode} steps")
        
        env.close()
        return episode_data
    
    def run_experiments(self, environments, episodes_per_env=10):
        """
        Run experiments across multiple environments.
        
        Args:
            environments: List of environment keys to test
            episodes_per_env: Number of episodes to run per environment
        """
        print(f"🧪 Starting experiments on {len(environments)} environments")
        print(f"Episodes per environment: {episodes_per_env}")
        print(f"Total episodes: {len(environments) * episodes_per_env}")
        print(f"Sleep duration: {self.sleep_duration}s per LLM call")
        
        total_episodes = 0
        
        for env_key in environments:
            print(f"\n📋 Testing environment: {env_key}")
            env_results = []
            
            for episode_id in range(episodes_per_env):
                episode_data = self.play_episode(env_key, episode_id, verbose=False)
                env_results.append(episode_data)
                total_episodes += 1
                
                # Print progress
                success_indicator = "✅" if episode_data['success'] else "❌"
                print(f"  Episode {episode_id}: {success_indicator} "
                      f"({episode_data['steps']} steps, {episode_data['termination_reason']})")
            
            # Calculate environment-specific stats
            successes = sum(1 for ep in env_results if ep['success'])
            success_rate = successes / episodes_per_env
            avg_steps = np.mean([ep['steps'] for ep in env_results])
            avg_successful_steps = np.mean([ep['steps'] for ep in env_results if ep['success']]) if successes > 0 else 0
            
            env_summary = {
                'env_key': env_key,
                'total_episodes': episodes_per_env,
                'successes': successes,
                'success_rate': success_rate,
                'avg_steps': avg_steps,
                'avg_successful_steps': avg_successful_steps,
                'episodes': env_results
            }
            
            self.results['episodes'].extend(env_results)
            self.results['summary'][env_key] = env_summary
            
            print(f"  📊 {env_key} Summary: {successes}/{episodes_per_env} success "
                  f"({success_rate:.1%}), avg steps: {avg_steps:.1f}")
        
        # Calculate overall stats
        total_successes = sum(1 for ep in self.results['episodes'] if ep['success'])
        overall_success_rate = total_successes / total_episodes
        overall_avg_steps = np.mean([ep['steps'] for ep in self.results['episodes']])
        
        self.results['overall'] = {
            'total_episodes': total_episodes,
            'total_successes': total_successes,
            'overall_success_rate': overall_success_rate,
            'overall_avg_steps': overall_avg_steps
        }
        
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"Total episodes: {total_episodes}")
        print(f"Total successes: {total_successes}")
        print(f"Overall success rate: {overall_success_rate:.1%}")
        print(f"Average steps per episode: {overall_avg_steps:.1f}")
        
        return self.results
    
    def save_results(self, filename=None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_episode_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")
        return filename
    
    def print_detailed_analysis(self):
        """Print detailed analysis of results."""
        print("\n" + "="*60)
        print("📈 DETAILED ANALYSIS")
        print("="*60)
        
        for env_key, env_data in self.results['summary'].items():
            print(f"\n🔍 {env_key}:")
            print(f"  Success Rate: {env_data['success_rate']:.1%} ({env_data['successes']}/{env_data['total_episodes']})")
            print(f"  Average Steps: {env_data['avg_steps']:.1f}")
            if env_data['avg_successful_steps'] > 0:
                print(f"  Average Steps (Successful): {env_data['avg_successful_steps']:.1f}")
            
            # Analyze termination reasons
            termination_counts = {}
            for ep in env_data['episodes']:
                reason = ep['termination_reason']
                termination_counts[reason] = termination_counts.get(reason, 0) + 1
            
            print(f"  Termination Reasons:")
            for reason, count in termination_counts.items():
                print(f"    {reason}: {count}")
        
        # Action analysis
        print(f"\n🎮 ACTION ANALYSIS:")
        all_actions = []
        for ep in self.results['episodes']:
            all_actions.extend(ep['actions'])
        
        if all_actions:
            action_names = {0: 'Turn left', 1: 'Turn right', 2: 'Move forward', 
                          3: 'Pick up', 4: 'Drop', 5: 'Toggle', 6: 'Done'}
            
            action_counts = {}
            for action in all_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            total_actions = len(all_actions)
            for action_id, count in sorted(action_counts.items()):
                percentage = count / total_actions * 100
                action_name = action_names.get(action_id, f'Unknown ({action_id})')
                print(f"  {action_name}: {count} ({percentage:.1f}%)")

def main():
    """Main function to run the experiments."""
    # Define environments to test
    environments = [
        "BabyAI-GoToObj-v0",
        "BabyAI-OpenDoor-v0", 
        "BabyAI-PickupLoc-v0"
    ]
    
    # Initialize player with 2-second sleep to avoid rate limits
    player = LLMEpisodePlayer(sleep_duration=2.0, max_steps_per_episode=50)
    
    # Run experiments
    results = player.run_experiments(environments, episodes_per_env=10)
    
    # Print detailed analysis
    player.print_detailed_analysis()
    
    # Save results
    filename = player.save_results()
    
    print(f"\n✨ Experiment completed! Results saved to {filename}")

if __name__ == "__main__":
    main() 