import dspy
import gymnasium as gym
import random
from minigrid.wrappers import FullyObsWrapper
from minigrid.utils.baby_ai_bot import BabyAIBot

from utils.observation_encoder import ObservationEncoder
from utils.env import make_env
from utils.dspy_signature import configure_llm, SubgoalPredictor
from utils.config_task_desc import task_desc
from dotenv import load_dotenv
from utils.hint_wrapper import HintWrapper

load_dotenv()

# Configure the LLM
configure_llm()
llm_agent = SubgoalPredictor()
encoder = ObservationEncoder()

# List of environments to collect from
environments = [
    "BabyAI-GoToObj-v0",
    "BabyAI-OpenDoor-v0", 
    "BabyAI-PickupLoc-v0"
]

examples_per_env = 10
total_examples = len(environments) * examples_per_env
llm_query_id = 1
episode_num = 1

with open('llm_eval.txt', 'w') as f:
    f.write(f"Collecting {examples_per_env} examples from each environment ({total_examples} total)\n")
    f.write(f"Environments: {environments}\n")
    f.write("=" * 60 + "\n\n")
    
    # Collect examples from each environment sequentially
    for env_idx, env_key in enumerate(environments):
        env_examples_collected = 0
        f.write(f"\n--- Starting {env_key} (Target: {examples_per_env} examples) ---\n\n")
        print(f"\n=== Collecting from {env_key} (Target: {examples_per_env} examples) ===")
        
        while env_examples_collected < examples_per_env:
            print(f"Starting Episode {episode_num} ({env_key}) - collected {env_examples_collected}/{examples_per_env} from this env")
            
            # Create fresh environment for each episode
            env = make_env(env_key=env_key)
            env = FullyObsWrapper(env)
            env = HintWrapper(env, hint_type="action", hint_source="dspy", hint_frequency=5)
            
            obs, _ = env.reset()
            
            # Initialize BabyAI bot
            bot = BabyAIBot(env.unwrapped)
            
            step_count = 0
            done = False
            episode_queries = 0
            
            f.write(f"Episode {episode_num} ({env_key}):\n")
            f.write("-" * 40 + "\n")
            
            while not done and env_examples_collected < examples_per_env:
                # Get action from BabyAI bot
                action = bot.replan()
                
                # Check if we should query LLM (every 5 steps)
                hint_enabled = (step_count % 5 == 0)
                
                if hint_enabled:
                    # Encode current observation
                    all_encodings = encoder.encode_all(obs)
                    current_state = all_encodings['ascii']
                    
                    # Query LLM
                    response = llm_agent(task_description=task_desc, current_state=current_state, previous_actions=[])
                    
                    # Save to file with ID
                    f.write(f'ID: {llm_query_id}\n')
                    f.write(f'Environment: {env_key}\n')
                    f.write(f'Episode: {episode_num}, Step: {step_count}\n')
                    f.write(f'Observation:\n{current_state}\n')
                    f.write(f'LLM Response: {response}\n')
                    f.write('=' * 50 + '\n\n')
                    
                    print(f'  LLM Query ID {llm_query_id} at step {step_count}')
                    llm_query_id += 1
                    episode_queries += 1
                    env_examples_collected += 1
                
                # Take action in environment
                obs, reward, term, trunc, _ = env.step(action)
                step_count += 1
                
                # Check if done
                if term or trunc or step_count > 100:
                    done = True
            
            env.close()
            print(f"  Episode {episode_num} completed in {step_count} steps with {episode_queries} LLM queries")
            episode_num += 1
        
        print(f"✅ Completed {env_key}: {env_examples_collected}/{examples_per_env} examples collected")

print(f'\nCollection completed! Collected {llm_query_id - 1} examples across {episode_num - 1} episodes')
print(f'Distribution: {examples_per_env} examples from each of {len(environments)} environments')
