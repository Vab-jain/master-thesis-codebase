import gymnasium as gym
import minigrid
import numpy as np
import sys
import os

# Add utils to path to import BabyAI Bot
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from baby_ai_bot import BabyAIBot

def format_observation_example(obs, action, step_num, description):
    """Format an observation and action as a training example."""
    
    # Convert numpy arrays to lists for better readability
    image_list = obs['image'].tolist()
    
    # Action mapping
    action_names = {
        0: "left",
        1: "right", 
        2: "forward",
        3: "pickup",
        4: "drop",
        5: "toggle",
        6: "done"
    }
    
    action_name = action_names.get(action, f"unknown({action})")
    
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

def analyze_observation(obs, step_num):
    """Analyze what's in the observation and create a description."""
    
    grid = obs['image']
    height, width = grid.shape[:2]
    
    # Find different objects in the grid
    objects_found = []
    agent_pos = None
    
    for y in range(height):
        for x in range(width):
            cell = grid[y][x]
            obj_type = cell[0]
            obj_color = cell[1] if len(cell) > 1 else 0
            
            if obj_type == 10:  # Agent
                agent_pos = (x, y)
                objects_found.append(f"Agent at ({x},{y})")
            elif obj_type == 5:  # Key
                objects_found.append(f"Key at ({x},{y})")
            elif obj_type == 4:  # Door
                objects_found.append(f"Door at ({x},{y})")
            elif obj_type == 8:  # Goal
                objects_found.append(f"Goal at ({x},{y})")
            elif obj_type == 2:  # Wall
                continue  # Don't mention walls unless relevant
            elif obj_type == 1:  # Floor
                continue  # Don't mention floors
            elif obj_type != 0:  # Unknown object
                objects_found.append(f"Object{obj_type} at ({x},{y})")
    
    direction_names = ["right", "down", "left", "up"]
    direction_name = direction_names[obs['direction']]
    
    description = f"Agent facing {direction_name}. " + "; ".join(objects_found)
    return description

def generate_examples_from_bot():
    """Generate training examples using the BabyAI Bot."""
    
    print("Generating LLM Training Examples using BabyAI Bot")
    print("=" * 60)
    
    # Create BabyAI environment (which has instructions and works with BabyAI Bot)
    env = gym.make("BabyAI-UnlockPickup-v0")
    obs_tuple = env.reset()
    obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
    
    # Create bot
    bot = BabyAIBot(env)
    
    examples = []
    step_count = 0
    max_examples = 10
    
    print(f"Initial mission: {obs['mission']}")
    print(f"Environment: BabyAI-UnlockPickup-v0")
    print()
    
    while step_count < max_examples:
        try:
            # Get action from expert bot
            action, subgoal_name = bot.replan()
            
            # Create description of current state
            description = analyze_observation(obs, step_count + 1)
            
            # Format as training example
            example = format_observation_example(obs, action, step_count + 1, description)
            examples.append(example)
            
            print(f"Step {step_count + 1}: {subgoal_name} -> {['left','right','forward','pickup','drop','toggle','done'][action]}")
            print(f"Description: {description}")
            
            # Take the action
            step_result = env.step(action)
            obs_tuple, reward, done, truncated, info = step_result
            obs = obs_tuple if isinstance(obs_tuple, dict) else obs_tuple
            done = done or truncated
            
            step_count += 1
            
            if done:
                if reward > 0:
                    print(f"\n✅ Mission completed successfully in {step_count} steps!")
                else:
                    print(f"\n❌ Mission failed after {step_count} steps")
                break
                
        except Exception as e:
            print(f"Error at step {step_count + 1}: {e}")
            break
    
    env.close()
    return examples

def create_llm_prompt_template(examples):
    """Create a complete prompt template for LLM training."""
    
    template = '''The environment is BabyAI-UnlockPickup-v0. The agent must pick up the key, unlock the door, and reach the goal.

The observation format:
- "image": A 7x7x3 array representing the agent's view. Each cell [obj_type, color, state]:
  * obj_type: 0=empty, 1=floor, 2=wall, 4=door, 5=key, 8=goal, 10=agent
  * color: 0=red, 1=green, 2=blue, 3=purple, 4=yellow, 5=grey
  * state: For doors: 0=closed, 1=open. For keys: usually 0.
- "direction": Agent's facing direction (0=right, 1=down, 2=left, 3=up)
- "mission": Text description of the task

Action space:
- "left": Turn left (counter-clockwise)
- "right": Turn right (clockwise)  
- "forward": Move forward one step
- "pickup": Pick up object in front
- "drop": Drop carried object
- "toggle": Open/close door
- "done": Signal task completion

Here are examples of observations and correct actions:
'''
    
    for example in examples:
        template += example
    
    template += '''
Using this format and these examples, write a Python function:

def planner(observation):
    """
    Plan the next action based on the observation.
    
    Args:
        observation (dict): Contains 'image', 'direction', and 'mission'
        
    Returns:
        str: One of 'left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done'
    """
    # Your implementation here
    pass
'''
    
    return template

def main():
    """Main function to generate examples and create LLM prompt."""
    
    # Generate examples using BabyAI Bot
    examples = generate_examples_from_bot()
    
    # Create complete prompt template
    prompt_template = create_llm_prompt_template(examples)
    
    # Save to file
    output_file = "llm_training_prompt.txt"
    with open(output_file, 'w') as f:
        f.write(prompt_template)
    
    print(f"\n📝 Complete LLM training prompt saved to: {output_file}")
    print(f"Generated {len(examples)} training examples")
    
    # Also create a simplified version for direct use
    simple_examples_file = "training_examples.txt"
    with open(simple_examples_file, 'w') as f:
        f.write("# BabyAI Bot Training Examples\n")
        f.write("# Generated from BabyAI-UnlockPickup-v0\n\n")
        for example in examples:
            f.write(example)
    
    print(f"📋 Raw examples saved to: {simple_examples_file}")
    
    return prompt_template, examples

if __name__ == "__main__":
    main() 