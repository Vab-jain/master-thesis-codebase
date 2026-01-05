import gymnasium as gym
import minigrid
import numpy as np
from generated_planner import planner
import time

def evaluate_planner(env_name="MiniGrid-Empty-5x5-v0", num_episodes=100, max_steps=100, render=False):
    """
    Evaluate the generated planner on the specified MiniGrid environment.
    
    Args:
        env_name: Name of the MiniGrid environment
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render: Whether to render the environment
    
    Returns:
        dict: Evaluation results including success rate, average steps, etc.
    """
    
    # Create environment
    env = gym.make(env_name)
    
    # Track results
    successes = 0
    total_steps = 0
    episode_lengths = []
    failed_episodes = []
    
    print(f"Evaluating planner on {env_name} for {num_episodes} episodes...")
    print(f"Max steps per episode: {max_steps}")
    print("-" * 50)
    
    for episode in range(num_episodes):
        obs_tuple = env.reset()
        obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple  # Extract obs from tuple
        done = False
        steps = 0
        
        if render:
            env.render()
            time.sleep(0.1)
        
        while not done and steps < max_steps:
            try:
                # Get action from planner
                action = planner(obs)
                
                # Ensure action is valid
                if not isinstance(action, int) or action < 0 or action >= env.action_space.n:
                    print(f"Warning: Invalid action {action} from planner, using forward (2)")
                    action = 2
                
                # Take step
                obs_tuple, reward, done, truncated, info = env.step(action)
                obs = obs_tuple if isinstance(obs_tuple, dict) else obs_tuple
                done = done or truncated  # Handle both done and truncated flags
                steps += 1
                total_steps += 1
                
                if render:
                    env.render()
                    time.sleep(0.1)
                
                # Check if goal reached
                if done and reward > 0:
                    successes += 1
                    episode_lengths.append(steps)
                    print(f"Episode {episode + 1}: SUCCESS in {steps} steps")
                    break
                    
            except Exception as e:
                print(f"Error in episode {episode + 1}: {e}")
                failed_episodes.append(episode + 1)
                break
        
        if not done or (done and reward <= 0):
            print(f"Episode {episode + 1}: FAILED (steps: {steps})")
            episode_lengths.append(steps)
    
    env.close()
    
    # Calculate statistics
    success_rate = successes / num_episodes
    avg_steps = np.mean(episode_lengths) if episode_lengths else 0
    avg_steps_successful = np.mean([length for i, length in enumerate(episode_lengths) 
                                   if i + 1 not in failed_episodes and 
                                   episode_lengths[i] <= max_steps]) if successes > 0 else 0
    
    results = {
        "environment": env_name,
        "num_episodes": num_episodes,
        "max_steps_per_episode": max_steps,
        "successes": successes,
        "success_rate": success_rate,
        "total_steps": total_steps,
        "average_steps_per_episode": avg_steps,
        "average_steps_successful_episodes": avg_steps_successful,
        "failed_episodes": failed_episodes,
        "episode_lengths": episode_lengths
    }
    
    return results

def print_results(results):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*60)
    print("PLANNER EVALUATION RESULTS")
    print("="*60)
    print(f"Environment: {results['environment']}")
    print(f"Episodes: {results['num_episodes']}")
    print(f"Max steps per episode: {results['max_steps_per_episode']}")
    print("-" * 40)
    print(f"Successes: {results['successes']}/{results['num_episodes']}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Average steps per episode: {results['average_steps_per_episode']:.1f}")
    if results['successes'] > 0:
        print(f"Average steps (successful episodes): {results['average_steps_successful_episodes']:.1f}")
    print(f"Total steps: {results['total_steps']}")
    
    if results['failed_episodes']:
        print(f"Failed episodes due to errors: {len(results['failed_episodes'])}")
    
    print("="*60)

if __name__ == "__main__":
    # Run evaluation
    results = evaluate_planner(
        env_name="MiniGrid-Empty-5x5-v0",
        num_episodes=100,
        max_steps=100,
        render=False  # Set to True if you want to see the episodes
    )
    
    # Print results
    print_results(results)
    
    # Save results to file
    import json
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"planner_evaluation_results_{timestamp}.json"
    
    # Make results JSON serializable
    json_results = results.copy()
    json_results['episode_lengths'] = [int(x) for x in results['episode_lengths']]
    json_results['failed_episodes'] = [int(x) for x in results['failed_episodes']]
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Also create a summary report
    summary_file = f"evaluation_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("PLANNER EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Environment: {results['environment']}\n")
        f.write(f"Planner: Generated LLM Planner\n")
        f.write(f"Episodes evaluated: {results['num_episodes']}\n")
        f.write(f"Max steps per episode: {results['max_steps_per_episode']}\n\n")
        f.write("RESULTS:\n")
        f.write(f"Success rate: {results['success_rate']:.2%} ({results['successes']}/{results['num_episodes']})\n")
        f.write(f"Total steps: {results['total_steps']}\n")
        f.write(f"Average steps per episode: {results['average_steps_per_episode']:.1f}\n")
        if results['successes'] > 0:
            f.write(f"Average steps for successful episodes: {results['average_steps_successful_episodes']:.1f}\n")
        f.write(f"Episodes with errors: {len(results['failed_episodes'])}\n")
        
        f.write("\nCONCLUSION:\n")
        if results['success_rate'] == 0:
            f.write("The LLM-generated planner failed to solve any episodes in MiniGrid-Empty-5x5-v0.\n")
            f.write("This indicates that the planner does not understand the MiniGrid observation format\n")
            f.write("and coordinate system correctly, leading to ineffective navigation behavior.\n")
        else:
            f.write(f"The planner achieved a {results['success_rate']:.1%} success rate.\n")
    
    print(f"Summary report saved to: {summary_file}") 