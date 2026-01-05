#!/usr/bin/env python3
"""
Wrapper that provides hints from BabyAI bot (subgoals or primitive actions) at regular intervals.
This wrapper can be used to compare RL agents trained with and without hints.
Supports both DSPy models and BabyAI bot as hint sources.
Properly handles parallel environments with isolated state.
"""

import gymnasium as gym
import numpy as np
import dspy
import json
import os
import time
import sys
import random
from typing import Dict, Any, Optional, Union, List
from .observation_encoder import ObservationEncoder

# Import local BabyAI bot from utils
from .baby_ai_bot import BabyAIBot

class HintWrapper(gym.ObservationWrapper):
    """
    Wrapper that adds hints from BabyAI bot to observations at regular intervals.
    Properly handles parallel environments with isolated state per environment.
    
    Args:
        env: The environment to wrap
        hint_type: Type of hint ("subgoal" or "action")
        hint_frequency: How often to provide hints (every k steps)
        hint_source: Source of hints ("dspy" or "babyai_bot")
        hint_model_path: Path to the fine-tuned DSPy model (optional, only for "dspy" source)
        encoding_type: Type of text encoding to use ("natural", "ascii", "tuples", "relative")
        hint_probability: Probability of providing a hint when it's time (default: 1.0)
    """
    
    def __init__(self, 
                 env: gym.Env, 
                 hint_type: str = "subgoal",
                 hint_frequency: int = 5,
                 hint_source: str = "dspy",
                 hint_model_path: Optional[str] = None,
                 encoding_type: str = "ascii",
                 hint_probability: float = 1.0,
                 hint_stop_percentage: float = 1.0):
        super().__init__(env)
        
        # Validate DSPy compatibility
        if hint_source == "dspy" and hint_type == "subgoal":
            raise ValueError(
                "DSPy hint source only supports hint_type='action'. "
                "The refactored DSPy signature only outputs primitive_action, not subgoals. "
                "Use hint_type='action' with hint_source='dspy', or use hint_source='babyai_bot' for subgoals."
            )
        
        self.hint_type = hint_type
        self.hint_frequency = hint_frequency
        self.hint_source = hint_source
        self.encoding_type = encoding_type
        self.hint_probability = hint_probability
        self.hint_stop_percentage = hint_stop_percentage
        self.hints_disabled = False  # Flag to disable hints after reaching stop percentage
        self.step_count = 0
        
        # Track previous actions for this environment instance (last 5 actions)
        self.previous_actions_history: List[str] = []  # Store as formatted strings
        self.max_action_history = 5
        self.step_counter = 0  # Track total steps for formatting
        
        # Create unique identifier for this environment instance
        self.env_id = id(env)
        # Use a simpler identifier to avoid issues with uninitialized environment state
        self.env_instance_name = f"env_{self.env_id}_{id(self)}"
        
        # Initialize the hint source
        self._init_hint_source(hint_model_path)
        
        # Initialize observation encoder (only needed for DSPy)
        if self.hint_source == "dspy":
            self.encoder = ObservationEncoder()
        
        # Update observation space to include hint
        self._update_observation_space()
        
        print(f"🔧 HintWrapper initialized for {self.env_instance_name} with {hint_source} hints")
    
    def _init_hint_source(self, hint_model_path: Optional[str]):
        """Initialize the hint source (DSPy model or BabyAI bot)."""
        if self.hint_source == "dspy":
            # Import DSPy-related modules only when needed
            from utils.dspy_signature import SubgoalPredictor, configure_llm
            
            # Configure LLM (this sets up the DSPy LM, not LiteLLM)
            configure_llm()
            
            # Initialize predictor (no mode parameter needed in the refactored version)
            self.predictor = SubgoalPredictor()
            
            # Load fine-tuned model if provided
            if hint_model_path and os.path.exists(hint_model_path):
                try:
                    self.predictor.load(hint_model_path)
                    print(f"✅ Loaded fine-tuned hint model from: {hint_model_path}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load hint model from {hint_model_path}: {e}")
                    print("Using base model instead.")
        
        elif self.hint_source == "babyai_bot":
            # Initialize BabyAI bot as oracle - will be created per environment instance
            # This ensures each parallel environment has its own isolated bot state
            self.babyai_bot = None
            self._bot_initialized = False
            print(f"🤖 BabyAI bot will be initialized for {self.env_instance_name} after environment reset")
        
        else:
            raise ValueError(f"Unknown hint source: {self.hint_source}. Use 'dspy' or 'babyai_bot'")

    def _init_babyai_bot(self):
        """Initialize BabyAI bot with proper isolation for this environment instance."""
        try:
            # Create a fresh BabyAI bot instance for this specific environment
            self.babyai_bot = BabyAIBot(self.env)
            self._bot_initialized = True
            
            # Ensure the bot processes the current observation
            self.babyai_bot._process_obs()
            self.babyai_bot._remember_current_state()
            
            print(f"✅ BabyAI bot initialized successfully for {self.env_instance_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing BabyAI bot for {self.env_instance_name}: {e}")
            print("Falling back to DSPy model")
            
            # Fallback to DSPy
            self.hint_source = "dspy"
            self._init_hint_source(None)
            return False

    def _update_observation_space(self):
        """Update observation space to include hint information."""
        from gymnasium.spaces import Dict, Box, Discrete
        
        # Get the original observation space
        original_space = self.env.observation_space
        
        # Create hint space
        if self.hint_type == "subgoal":
            # Subgoal is a string, we'll encode it as integers
            # Common subgoals: CloseSubgoal, OpenSubgoal, DropSubgoal, PickupSubgoal, 
            # GoNextToSubgoal, ExploreSubgoal, done, none
            hint_space = Discrete(8)  # 8 possible subgoals
        else:  # action
            # Primitive actions: 0-6 (turn left, turn right, move forward, pick up, drop, toggle, done)
            hint_space = Discrete(7)
        
        # Create availability space (whether hint is available)
        availability_space = Discrete(2)  # 0 = no hint, 1 = hint available
        
        # Combine with original observation space
        if isinstance(original_space, gym.spaces.Dict):
            self.observation_space = Dict({
                **original_space.spaces,
                'hint': hint_space,
                'hint_available': availability_space
            })
        else:
            # If original space is not a dict, create a new dict
            self.observation_space = Dict({
                'observation': original_space,
                'hint': hint_space,
                'hint_available': availability_space
            })

    def _get_previous_actions_for_dspy(self) -> List[str]:
        """
        Get the previous actions formatted for DSPy predictor.
        Returns up to the last 5 previous actions (already formatted as strings).
        Format: ['step-1: 2', 'step-2: 0', 'step-3: 2']
        If no actions taken yet, returns empty list.
        """
        if not self.previous_actions_history:
            return []  # No actions taken yet
        
        # Actions are already formatted, just return the last 5 (or fewer)
        return self.previous_actions_history[-5:]

    def _get_hint_from_dspy(self, obs: Dict[str, Any]) -> tuple[int, bool]:
        """Get hint from DSPy model with retry mechanism and exponential backoff."""
        max_retries = 5
        base_delay = 10.0  # Start with 10 seconds delay for rate limits (minute-based limits)
        max_delay = 300.0  # Maximum delay of 5 minutes
        
        # Pre-compute inputs once to avoid repeated computation
        current_state = self.encoder.encode_all(obs)[self.encoding_type]
        from utils.config_task_desc import task_desc
        task_description = task_desc if task_desc else obs.get('mission', '')
        
        # Get previous actions for DSPy (already formatted, just join with commas)
        previous_actions_list = self._get_previous_actions_for_dspy()
        previous_actions = ', '.join(previous_actions_list) if previous_actions_list else ""
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt without delay
            try:
                # Get prediction from model using the updated signature
                prediction = self.predictor(
                    task_description=task_description,
                    current_state=current_state,
                    previous_actions=previous_actions
                )
                
                # Extract hint value - DSPy only supports action hints now
                # (subgoal + dspy combination is prevented in __init__)
                action_val = prediction.primitive_action
                # Ensure it's an integer
                try:
                    hint_value = int(action_val)
                    # Validate action range (0-6 for actions)
                    if not (0 <= hint_value <= 6):
                        hint_value = 6  # Default to "done" if out of range
                except (ValueError, TypeError):
                    hint_value = 6  # Default to "done" instead of 0
                
                # Success! Return the hint
                if attempt > 0:
                    print(f"✅ DSPy hint generation succeeded after {attempt} retries for {self.env_instance_name}")
                return hint_value, True
                
            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit = any(keyword in error_msg for keyword in [
                    '429', 'rate limit', 'too many requests', 'quota exceeded'
                ])
                
                if attempt < max_retries and is_rate_limit:
                    # Calculate exponential backoff delay with jitter for rate limits
                    # Start at 10s minimum since rate limits are minute-based
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = delay * 0.1 * np.random.random()  # Add 10% jitter
                    total_delay = delay + jitter
                    
                    print(f"⚠️  DSPy rate limit hit for {self.env_instance_name} (attempt {attempt + 1}/{max_retries + 1}). "
                          f"Retrying in {total_delay:.1f}s... Error: {e}")
                    time.sleep(total_delay)
                    continue
                elif attempt < max_retries:
                    # For non-rate-limit errors, use shorter delay
                    delay = min(2.0 * (attempt + 1), 10.0)  # 2s, 4s, 6s, 8s, 10s
                    print(f"⚠️  DSPy error for {self.env_instance_name} (attempt {attempt + 1}/{max_retries + 1}). "
                          f"Retrying in {delay:.1f}s... Error: {e}")
                    time.sleep(delay)
                    continue
                else:
                    # Final attempt failed
                    print(f"❌ DSPy hint generation failed after {max_retries} retries for {self.env_instance_name}. "
                          f"Final error: {e}\n⚠️  Falling back to BabyAI bot hints.")
                    # Switch to BabyAI bot as fallback
                    self.hint_source = "babyai_bot"
                    # Ensure BabyAI bot is initialized
                    if not hasattr(self, "_bot_initialized") or not self._bot_initialized or self.babyai_bot is None:
                        # Attempt to initialize BabyAI bot; ignore return value because we will call it right away
                        self._init_babyai_bot()
                    # Try getting a hint from BabyAI bot; if that fails return neutral action
                    try:
                        return self._get_hint_from_babyai_bot(obs)
                    except Exception as fallback_err:
                        print(f"🚫 Fallback BabyAI bot hint failed for {self.env_instance_name}: {fallback_err}")
                        # Return neutral action when all else fails
                        return 6, False
            
        # All retries exhausted but this line should not be reached because of fallback above.
        # Added as a safety net.
        print(f"🚫 DSPy fallback mechanism reached unexpected state for {self.env_instance_name}. Returning neutral action.")
        return 6, False  # 6 = "done"

    def _get_hint_from_babyai_bot(self, obs: Dict[str, Any]) -> tuple[int, bool]:
        """Get hint from BabyAI bot oracle with proper state isolation."""
        try:
            # Ensure bot is initialized for this environment instance
            if not self._bot_initialized or self.babyai_bot is None:
                if not self._init_babyai_bot():
                    # Fallback to neutral hint if initialization failed
                    if self.hint_type == "subgoal":
                        return 7, False  # 7 = "none" for subgoals
                    else:  # action
                        return 6, False  # 6 = "done" (most neutral action)
            
            # Ensure bot state is synchronized with current environment state
            self.babyai_bot._process_obs()
            
            # Get the last action (extract integer from formatted string, or use None for the first step)
            last_action = None
            if self.previous_actions_history:
                # Extract action number from formatted string like "step-3: 2"
                last_action_str = self.previous_actions_history[-1]
                try:
                    last_action = int(last_action_str.split(': ')[1])
                except (IndexError, ValueError):
                    last_action = None
            
            # Get hint from BabyAI bot using replan method
            # This returns (action, subgoal_name) tuple as seen in sand_gt_data_collection.py
            action, subgoal_name = self.babyai_bot.replan(last_action)
            
            # Extract hint value based on type
            if self.hint_type == "subgoal":
                # Convert subgoal name to integer encoding
                # Note: subgoal_name is the CURRENT subgoal being pursued (strategic guidance)
                hint_value = self._subgoal_to_int(subgoal_name)
            else:  # action
                # Convert action to integer (actions are already integers from BabyAI bot)
                # This is the NEXT action the bot suggests taking - direct action guidance
                hint_value = int(action)
            
            return hint_value, True
            
        except Exception as e:
            print(f"⚠️  Warning: Error getting hint from BabyAI bot for {self.env_instance_name}: {e}")
            # Use neutral "none" value instead of 0 to avoid confusion with actual hints
            if self.hint_type == "subgoal":
                return 7, False  # 7 = "none" for subgoals
            else:  # action
                return 6, False  # 6 = "done" (most neutral action)
    
    def _get_hint(self, obs: Dict[str, Any]) -> tuple[int, bool]:
        """
        Get hint from the configured source.
        
        Returns:
            tuple: (hint_value, hint_available)
        """
        if self.hints_disabled:
            return 6, False # 6 = "done" (most neutral action)

        if self.hint_source == "dspy":
            return self._get_hint_from_dspy(obs)
        elif self.hint_source == "babyai_bot":
            return self._get_hint_from_babyai_bot(obs)
        else:
            # Unknown hint source - use neutral "none" value instead of 0
            if self.hint_type == "subgoal":
                return 7, False  # 7 = "none" for subgoals
            else:  # action
                return 6, False  # 6 = "done" (most neutral action)
    
    def _subgoal_to_int(self, subgoal: str) -> int:
        """Convert subgoal string to integer encoding."""
        subgoal_map = {
            "CloseSubgoal": 0,
            "OpenSubgoal": 1,
            "DropSubgoal": 2,
            "PickupSubgoal": 3,
            "GoNextToSubgoal": 4,
            "ExploreSubgoal": 5,
            "done": 6,
            "none": 7
        }
        return subgoal_map.get(subgoal, 7)  # Default to "none"
    
    def _int_to_subgoal(self, hint_int: int) -> str:
        """Convert integer encoding back to subgoal string."""
        subgoal_map = {
            0: "CloseSubgoal",
            1: "OpenSubgoal", 
            2: "DropSubgoal",
            3: "PickupSubgoal",
            4: "GoNextToSubgoal",
            5: "ExploreSubgoal",
            6: "done",
            7: "none"
        }
        return subgoal_map.get(hint_int, "none")
    
    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Add hint to observation."""
        self.step_count += 1
        
        # Determine if we should provide a hint
        should_hint = (self.step_count % self.hint_frequency == 0 and 
                      np.random.random() < self.hint_probability)
        
        if should_hint:
            hint_value, hint_available = self._get_hint(obs)
        else:
            # Use "none" value (7 for subgoals) instead of 0 to avoid confusion with actual hints
            # For subgoals: 7="none", for actions: 6="done" (close to neutral)
            if self.hint_type == "subgoal":
                hint_value = 7  # "none" 
            else:  # action
                hint_value = 6  # "done" (most neutral action)
            hint_available = False
        
        # Add hint to observation
        if isinstance(obs, dict):
            obs['hint'] = hint_value
            obs['hint_available'] = int(hint_available)
            return obs
        else:
            # If obs is not a dict, wrap it
            return {
                'observation': obs,
                'hint': hint_value,
                'hint_available': int(hint_available)
            }
    
    def reset(self, **kwargs):
        """Reset the wrapper and environment with proper state isolation."""
        self.step_count = 0
        self.step_counter = 0  # Reset step counter for action formatting
        
        # Reset action history for this environment instance
        self.previous_actions_history = []
        
        # Reset environment-specific state
        if hasattr(self, '_last_action'):
            delattr(self, '_last_action')
        
        obs, info = self.env.reset(**kwargs)
        
        # Initialize or re-initialize BabyAI bot for this environment instance
        if self.hint_source == "babyai_bot":
            # Always re-initialize the bot on reset to ensure clean state
            self.babyai_bot = None
            self._bot_initialized = False
            
            if not self._init_babyai_bot():
                # If initialization fails, the source will have been changed to dspy
                pass
        
        return self.observation(obs), info

    def step(self, action):
        """Override step to track actions and ensure proper synchronization with BabyAI bot."""
        # Increment step counter
        self.step_counter += 1
        
        # Add the action to our history with step formatting
        formatted_action = f'step-{self.step_counter}: {action}'
        self.previous_actions_history.append(formatted_action)
        
        # Keep only the last 5 actions
        if len(self.previous_actions_history) > self.max_action_history:
            self.previous_actions_history = self.previous_actions_history[-self.max_action_history:]
        
        # Take the step in the environment
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Update BabyAI bot state if using babyai_bot hints
        if self.hint_source == "babyai_bot" and self._bot_initialized and self.babyai_bot is not None:
            try:
                # Ensure bot state stays synchronized with environment
                self.babyai_bot._process_obs()
                self.babyai_bot._remember_current_state()
            except Exception as e:
                print(f"⚠️  Warning: Could not update BabyAI bot state for {self.env_instance_name}: {e}")
        
        # Process the observation (add hints)
        return self.observation(obs), reward, done, truncated, info

    def disable_hints(self):
        """Disable hints (called externally when training reaches stop percentage)."""
        if not self.hints_disabled:
            print(f"🚫 Hints disabled for {self.env_instance_name} after reaching {self.hint_stop_percentage:.0%} of training")
            self.hints_disabled = True
    
    def enable_hints(self):
        """Re-enable hints if needed."""
        self.hints_disabled = False

def create_hint_env(env_id: str, 
                   hint_type: str = "subgoal",
                   hint_frequency: int = 5,
                   hint_source: str = "dspy",
                   hint_model_path: Optional[str] = None,
                   encoding_type: str = "ascii",
                   hint_probability: float = 1.0,
                   hint_stop_percentage: float = 1.0,
                   obs_type: str = "multi") -> gym.Env:
    """
    Create an environment with hints.
    
    Args:
        env_id: Environment ID
        hint_type: Type of hint ("subgoal" or "action")
        hint_frequency: How often to provide hints (every k steps)
        hint_source: Source of hints ("dspy" or "babyai_bot")
        hint_model_path: Path to the fine-tuned DSPy model (optional)
        encoding_type: Type of text encoding to use
        hint_probability: Probability of providing a hint when it's time
        hint_stop_percentage: Percentage of training steps after which hints are disabled
        obs_type: Observation type ("rgb", "multi", or "dict")
    
    Returns:
        Wrapped environment with hints
    """
    from .env import make_env
    from minigrid.wrappers import FullyObsWrapper
    env = make_env(env_id, wrappers=[FullyObsWrapper])
    env = HintWrapper(
        env=env,
        hint_type=hint_type,
        hint_frequency=hint_frequency,
        hint_source=hint_source,
        hint_model_path=hint_model_path,
        encoding_type=encoding_type,
        hint_probability=hint_probability,
        hint_stop_percentage=hint_stop_percentage
    )
    return env

if __name__ == "__main__":
    # Test the hint wrapper
    env = create_hint_env(
        "BabyAI-GoToObj-v0",
        hint_type="subgoal",
        hint_frequency=3,
        encoding_type="ascii",
        obs_type="multi"
    )
    
    print("Testing hint wrapper...")
    obs, info = env.reset()
    print(f"Initial observation keys: {obs.keys()}")
    print(f"Hint: {obs['hint']}, Available: {obs['hint_available']}")
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step+1}: Hint={obs['hint']}, Available={obs['hint_available']}")
        
        if done or truncated:
            break
    
    env.close()
    print("Test completed!") 