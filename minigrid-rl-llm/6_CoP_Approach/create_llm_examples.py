#!/usr/bin/env python3
"""
Comprehensive LLM Training Example Generator for MiniGrid Environments

This script generates concrete observation examples using expert policies
to train LLMs for MiniGrid navigation tasks.
"""

import gymnasium as gym
import minigrid
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Any

# Add utils to path for BabyAI Bot
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    from baby_ai_bot import BabyAIBot
    BABYAI_AVAILABLE = True
except ImportError:
    BABYAI_AVAILABLE = False
    print("BabyAI Bot not available - using simple policies only")

class ExampleGenerator:
    """Generates training examples for LLM planners."""
    
    def __init__(self):
        self.action_names = {
            0: "left",
            1: "right", 
            2: "forward",
            3: "pickup",
            4: "drop",
            5: "toggle",
            6: "done"
        }
        
    def smart_navigation_policy(self, obs: Dict) -> int:
        """Improved navigation policy with obstacle avoidance."""
        
        if isinstance(obs, tuple):
            obs = obs[0]
        
        image = obs['image']
        direction = obs['direction']
        
        # Find goal and agent positions
        goal_pos = None
        agent_pos = (3, 3)  # Agent is always at center of 7x7 view
        
        for y in range(7):
            for x in range(7):
                if image[y][x][0] == 8:  # Goal object
                    goal_pos = (x, y)
                    break
            if goal_pos:
                break
        
        if goal_pos is None:
            # No goal visible, explore by moving forward if possible
            if self._can_move_forward(image, direction):
                return 2  # forward
            else:
                return 1  # turn right to explore
        
        # Calculate relative position to goal
        dx = goal_pos[0] - agent_pos[0]
        dy = goal_pos[1] - agent_pos[1]
        
        # If at goal, signal done
        if dx == 0 and dy == 0:
            return 6  # done
        
        # Check if we can move forward toward the goal
        forward_pos = self._get_forward_pos(agent_pos, direction)
        can_move_forward = self._is_valid_position(image, forward_pos)
        
        # Determine desired direction based on Manhattan distance priority
        desired_dirs = []
        if dx != 0:
            desired_dirs.append(0 if dx > 0 else 2)  # right or left
        if dy != 0:
            desired_dirs.append(1 if dy > 0 else 3)  # down or up
        
        # Try to face a desired direction where we can actually move
        for desired_dir in desired_dirs:
            if direction == desired_dir and can_move_forward:
                # Check if moving forward gets us closer to goal
                new_dx = goal_pos[0] - forward_pos[0]
                new_dy = goal_pos[1] - forward_pos[1]
                if abs(new_dx) + abs(new_dy) < abs(dx) + abs(dy):
                    return 2  # forward
            elif direction == desired_dir:
                # Can't move forward in desired direction, turn to find alternative
                return 1  # turn right
        
        # Not facing desired direction, turn toward it
        if desired_dirs:
            target_dir = desired_dirs[0]
            turn_diff = (target_dir - direction) % 4
            if turn_diff <= 2:
                return 1 if turn_diff == 1 else 0  # right or left
            else:
                return 0  # left (shorter than turning right 3 times)
        
        # Default: turn right to explore
        return 1
    
    def _get_forward_pos(self, pos: Tuple[int, int], direction: int) -> Tuple[int, int]:
        """Get position one step forward from current position."""
        dir_vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, down, left, up
        dx, dy = dir_vectors[direction]
        return (pos[0] + dx, pos[1] + dy)
    
    def _can_move_forward(self, image: np.ndarray, direction: int) -> bool:
        """Check if agent can move forward without hitting obstacle."""
        agent_pos = (3, 3)
        forward_pos = self._get_forward_pos(agent_pos, direction)
        return self._is_valid_position(image, forward_pos)
    
    def _is_valid_position(self, image: np.ndarray, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (not wall or out of bounds)."""
        x, y = pos
        if x < 0 or x >= 7 or y < 0 or y >= 7:
            return False
        
        cell_type = image[y][x][0]
        # Valid if floor (1), goal (8), or empty (0)
        # Invalid if wall (2) or other obstacles
        return cell_type in [0, 1, 8]
    
    def analyze_observation(self, obs: Dict, policy_type: str = "smart") -> str:
        """Analyze observation and create description."""
        
        grid = obs['image']
        agent_pos = (3, 3)
        goal_pos = None
        
        # Find goal
        for y in range(7):
            for x in range(7):
                if grid[y][x][0] == 8:  # Goal
                    goal_pos = (x, y)
                    break
            if goal_pos:
                break
        
        direction_names = ["right", "down", "left", "up"]
        direction_name = direction_names[obs['direction']]
        
        # Check what's in front of the agent
        forward_pos = self._get_forward_pos(agent_pos, obs['direction'])
        can_move_forward = self._is_valid_position(grid, forward_pos)
        
        if goal_pos:
            dx = goal_pos[0] - agent_pos[0]
            dy = goal_pos[1] - agent_pos[1]
            distance = abs(dx) + abs(dy)
            
            direction_to_goal = ""
            if dx > 0:
                direction_to_goal += "right "
            elif dx < 0:
                direction_to_goal += "left "
            if dy > 0:
                direction_to_goal += "down"
            elif dy < 0:
                direction_to_goal += "up"
            
            obstacle_info = " (blocked ahead)" if not can_move_forward else " (clear ahead)"
            description = f"Agent facing {direction_name}. Goal {distance} steps {direction_to_goal.strip()}{obstacle_info}"
        else:
            obstacle_info = " (blocked ahead)" if not can_move_forward else " (clear ahead)"
            description = f"Agent facing {direction_name}. No goal visible{obstacle_info}"
        
        return description
    
    def format_example(self, obs: Dict, action: int, step_num: int, description: str) -> str:
        """Format observation and action as training example."""
        
        image_list = obs['image'].tolist()
        action_name = self.action_names.get(action, f"unknown({action})")
        
        example = f"""
Example {step_num}:
# {description}
observation = {{
    "image": {image_list},
    "direction": {obs['direction']},
    "mission": "{obs['mission']}"
}}
# Expected planner output: "{action_name}"
"""
        return example
    
    def generate_babyai_examples(self, env_name: str = "BabyAI-UnlockPickup-v0", max_examples: int = 15) -> List[str]:
        """Generate examples using BabyAI Bot (if available)."""
        
        if not BABYAI_AVAILABLE:
            print("BabyAI not available, skipping complex examples")
            return []
        
        print(f"Generating BabyAI examples for {env_name}")
        
        env = gym.make(env_name)
        obs_tuple = env.reset()
        obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
        
        bot = BabyAIBot(env)
        examples = []
        step_count = 0
        
        while step_count < max_examples:
            try:
                action, subgoal_name = bot.replan()
                description = self.analyze_observation(obs, "babyai") + f" (Subgoal: {subgoal_name})"
                
                example = self.format_example(obs, action, len(examples) + 1, description)
                examples.append(example)
                
                print(f"  Example {len(examples)}: {subgoal_name} -> {self.action_names[action]}")
                
                step_result = env.step(action)
                obs_tuple, reward, done, truncated, info = step_result
                obs = obs_tuple if isinstance(obs_tuple, dict) else obs_tuple
                done = done or truncated
                
                step_count += 1
                
                if done:
                    print(f"  Episode completed after {step_count} steps")
                    break
                    
            except Exception as e:
                print(f"  Error: {e}")
                break
        
        env.close()
        return examples
    
    def generate_simple_examples(self, env_name: str = "MiniGrid-Empty-5x5-v0", 
                                episodes: int = 2, max_steps: int = 15) -> List[str]:
        """Generate simple navigation examples."""
        
        print(f"Generating simple examples for {env_name}")
        
        env = gym.make(env_name)
        all_examples = []
        
        for episode in range(episodes):
            print(f"  Episode {episode + 1}:")
            obs_tuple = env.reset()
            obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
            
            step_count = 0
            episode_examples = 0
            
            while step_count < max_steps and episode_examples < 8:  # Limit examples per episode
                action = self.smart_navigation_policy(obs)
                description = self.analyze_observation(obs, "smart")
                
                example = self.format_example(obs, action, len(all_examples) + 1, description)
                all_examples.append(example)
                episode_examples += 1
                
                print(f"    Step {step_count + 1}: {description} -> {self.action_names[action]}")
                
                step_result = env.step(action)
                obs_tuple, reward, done, truncated, info = step_result
                obs = obs_tuple if isinstance(obs_tuple, dict) else obs_tuple
                done = done or truncated
                
                step_count += 1
                
                if done:
                    if reward > 0:
                        print(f"    ✅ Goal reached!")
                    break
        
        env.close()
        return all_examples
    
    def create_comprehensive_prompt(self, simple_examples: List[str], 
                                  babyai_examples: List[str] = None) -> str:
        """Create comprehensive training prompt with multiple environments."""
        
        if babyai_examples is None:
            babyai_examples = []
        
        template = '''# MiniGrid LLM Planner Training Examples

This dataset contains concrete examples of expert navigation behavior in MiniGrid environments. Use these examples to learn how to make optimal decisions based on partial observations.

## Observation Format
- **image**: 7x7x3 array representing agent's partial view of the environment
  - Each cell: [object_type, color, state]
  - Object types: 0=empty, 1=floor, 2=wall, 4=door, 5=key, 7=box, 8=goal, 10=agent
  - Colors: 0=red, 1=green, 2=blue, 3=purple, 4=yellow, 5=grey
- **direction**: Agent's facing direction (0=right, 1=down, 2=left, 3=up)  
- **mission**: Text description of the task

## Action Space
- **left**: Turn left (counter-clockwise)
- **right**: Turn right (clockwise)
- **forward**: Move forward one step
- **pickup**: Pick up object in front of agent
- **drop**: Drop carried object in front
- **toggle**: Open/close doors, activate switches
- **done**: Signal task completion

## Navigation Strategy
1. **Locate the goal** in the observation grid
2. **Check for obstacles** in the intended direction  
3. **Turn toward goal** if not facing the right direction
4. **Move forward** when path is clear and direction is correct
5. **Avoid walls** and navigate around obstacles
6. **Complete sub-tasks** like collecting keys or opening doors

'''

        if babyai_examples:
            template += "## Complex Task Examples (BabyAI)\n"
            template += "These examples show multi-step reasoning for tasks involving keys, doors, and objects:\n"
            for example in babyai_examples:
                template += example
        
        template += "\n## Navigation Examples (MiniGrid-Empty)\n"
        template += "These examples show basic navigation and obstacle avoidance:\n"
        for example in simple_examples:
            template += example
        
        template += '''

## Task: Implement the Planner Function

Using the examples above, implement a robust planner that can handle:
- Basic navigation to visible goals
- Obstacle avoidance (walls, other objects)
- Multi-step tasks (keys, doors) when applicable
- Efficient pathfinding with minimal turning

```python
def planner(observation):
    """
    Plan the next action based on the current observation.
    
    Args:
        observation (dict): Contains 'image', 'direction', and 'mission'
        
    Returns:
        str: One of 'left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done'
    """
    # Your implementation here
    # Consider: goal location, obstacles, agent direction, task requirements
    pass
```

## Key Implementation Tips:
1. **Parse the observation** to find goal position and obstacles
2. **Calculate optimal direction** using Manhattan distance
3. **Check for obstacles** before moving forward  
4. **Handle edge cases** like walls at environment boundaries
5. **Plan multi-step sequences** for complex tasks (keys → doors → goals)
6. **Use efficient turning** (shortest rotation to desired direction)
'''
        
        return template

def main():
    """Generate comprehensive LLM training examples."""
    
    print("=== MiniGrid LLM Training Example Generator ===\n")
    
    generator = ExampleGenerator()
    
    # Generate simple navigation examples
    simple_examples = generator.generate_simple_examples(
        env_name="MiniGrid-Empty-5x5-v0",
        episodes=2,
        max_steps=15
    )
    
    # Generate complex BabyAI examples (if available)
    babyai_examples = []
    if BABYAI_AVAILABLE:
        try:
            babyai_examples = generator.generate_babyai_examples(
                env_name="BabyAI-UnlockPickup-v0",
                max_examples=12
            )
        except Exception as e:
            print(f"BabyAI example generation failed: {e}")
    
    # Create comprehensive prompt
    comprehensive_prompt = generator.create_comprehensive_prompt(
        simple_examples, babyai_examples
    )
    
    # Save outputs
    files_created = []
    
    # Main comprehensive prompt
    main_file = "comprehensive_llm_training.txt"
    with open(main_file, 'w') as f:
        f.write(comprehensive_prompt)
    files_created.append(main_file)
    
    # Simple examples only
    if simple_examples:
        simple_file = "simple_navigation_examples.txt"
        with open(simple_file, 'w') as f:
            f.write("# Simple MiniGrid Navigation Examples\n\n")
            for example in simple_examples:
                f.write(example)
        files_created.append(simple_file)
    
    # BabyAI examples only
    if babyai_examples:
        babyai_file = "babyai_complex_examples.txt"
        with open(babyai_file, 'w') as f:
            f.write("# BabyAI Complex Task Examples\n\n")
            for example in babyai_examples:
                f.write(example)
        files_created.append(babyai_file)
    
    # Summary
    print(f"\n=== Generation Complete ===")
    print(f"📋 Simple examples: {len(simple_examples)}")
    print(f"🧠 BabyAI examples: {len(babyai_examples)}")
    print(f"📝 Total examples: {len(simple_examples) + len(babyai_examples)}")
    print(f"\nFiles created:")
    for file in files_created:
        print(f"  📄 {file}")
    
    return comprehensive_prompt

if __name__ == "__main__":
    main() 