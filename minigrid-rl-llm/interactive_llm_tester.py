#!/usr/bin/env python3
"""
Interactive LLM Tester for MinGrid Environments
Allows manual control of the agent while observing LLM hint suggestions.
Perfect for analyzing LLM reasoning and decision-making.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Optional, Dict, Any

# Add utils to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils.hint_wrapper import create_hint_env
from utils.observation_encoder import ObservationEncoder
from utils.env import make_env

# Action mappings for BabyAI environments
ACTION_NAMES = {
    0: "turn_left",
    1: "turn_right", 
    2: "move_forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done"
}

SUBGOAL_NAMES = {
    0: "CloseSubgoal",
    1: "OpenSubgoal",
    2: "DropSubgoal", 
    3: "PickupSubgoal",
    4: "GoNextToSubgoal",
    5: "ExploreSubgoal",
    6: "done",
    7: "none"
}

class InteractiveLLMTester:
    def __init__(self, 
                 env_id: str = "BabyAI-GoToObj-v0",
                 hint_type: str = "subgoal",
                 hint_source: str = "dspy", 
                 hint_model_path: Optional[str] = None,
                 encoding_type: str = "ascii",
                 hint_frequency: int = 1,
                 seed: Optional[int] = None,
                 render_mode: str = "rgb_array",
                 show_llm_details: bool = True):
        
        self.env_id = env_id
        self.hint_type = hint_type
        self.hint_source = hint_source
        self.encoding_type = encoding_type
        self.seed = seed
        
        # Create environment with hints
        print(f"🚀 Creating environment: {env_id}")
        print(f"   Hint type: {hint_type}")
        print(f"   Hint source: {hint_source}")
        print(f"   Encoding: {encoding_type}")
        print(f"   Model path: {hint_model_path or 'None (base model)'}")
        
        # First create the base environment with FullyObsWrapper for agent_pos
        from minigrid.wrappers import FullyObsWrapper
        base_env = make_env(env_id, render_mode=render_mode, wrappers=[FullyObsWrapper])
        
        # Then wrap it with the hint wrapper
        from utils.hint_wrapper import HintWrapper
        self.env = HintWrapper(
            env=base_env,
            hint_type=hint_type,
            hint_frequency=hint_frequency,  # Get hints every step
            hint_source=hint_source,
            hint_model_path=hint_model_path,
            encoding_type=encoding_type,
            hint_probability=1.0  # Always provide hints when available
        )
        
        # Set render mode
        if hasattr(self.env.unwrapped, 'render_mode'):
            self.env.unwrapped.render_mode = render_mode
        
        # Initialize observation encoder for display
        self.encoder = ObservationEncoder()
        
        # Statistics
        self.step_count = 0
        self.episode_count = 0
        self.session_log = []
        
        # Display options
        self.show_llm_details = show_llm_details  # Show LLM config and call history
        
        print("✅ Environment created successfully!")

    def display_observation(self, obs: Dict[str, Any], save_path: Optional[str] = None):
        """Display the current observation as an image with encoding overlay."""
        # Get the RGB array from environment
        try:
            img = self.env.get_wrapper_attr('render')()
        except:
            # Fallback: try direct render
            img = self.env.render()
        
        if img is None:
            print("⚠️  Could not render environment image")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left subplot: Environment image
        ax1.imshow(img)
        ax1.set_title(f"Environment: {self.env_id}\nStep: {self.step_count}, Episode: {self.episode_count + 1}")
        ax1.axis('off')
        
        # Right subplot: Text encoding
        ax2.axis('off')
        
        # Get encoded text representation
        base_obs = {k: v for k, v in obs.items() if k not in ['hint', 'hint_available']}
        encoded_text = self.encoder.encode_all(base_obs)[self.encoding_type]
        
        # Format text for display
        display_text = f"ENCODING TYPE: {self.encoding_type.upper()}\n"
        display_text += "=" * 40 + "\n\n"
        display_text += encoded_text
        
        ax2.text(0.05, 0.95, display_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Observation saved to: {save_path}")
        
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to ensure display

    def display_llm_details(self):
        """Display details about the current LLM configuration."""
        print("\n" + "🧠 LLM CONFIGURATION".center(80, "="))
        
        if self.hint_source == "dspy":
            try:
                import dspy
                # Get current LM configuration
                lm = dspy.settings.lm
                if lm:
                    print(f"LLM Provider: {type(lm).__name__}")
                    if hasattr(lm, 'model'):
                        print(f"Model: {lm.model}")
                    if hasattr(lm, 'api_base') and lm.api_base:
                        print(f"API Base: {lm.api_base}")
                    if hasattr(lm, 'max_tokens'):
                        print(f"Max Tokens: {lm.max_tokens}")
                    if hasattr(lm, 'temperature'):
                        print(f"Temperature: {lm.temperature}")
                else:
                    print("No LLM configured in DSPy settings")
                    
                # Check if there's a model loaded in the hint wrapper
                if hasattr(self.env, 'predictor') and hasattr(self.env.predictor, 'model'):
                    print(f"Predictor Model: {self.env.predictor.model}")
                    
            except Exception as e:
                print(f"Error getting LLM details: {e}")
        elif self.hint_source == "babyai_bot":
            print("Hint Source: BabyAI Bot (Oracle)")
            print("No LLM used - using deterministic bot logic")
        else:
            print(f"Unknown hint source: {self.hint_source}")
            
        print("=" * 80)

    def display_dspy_history(self):
        """Display the last DSPy call history if available."""
        if self.hint_source != "dspy":
            return
            
        try:
            import dspy
            print("\n" + "📞 LAST LLM CALL DETAILS".center(80, "="))
            
            # Get the last call from history
            history = dspy.inspect_history(1)
            
            if not history:
                print("No DSPy call history available")
                print("=" * 80)
                return
                
            last_call = history[-1] if history else None
            
            if last_call:
                print("🔍 Last LLM Interaction:")
                print("-" * 40)
                
                # Display prompt/input
                if 'prompt' in last_call:
                    prompt = last_call['prompt']
                    print(f"📝 PROMPT SENT TO LLM:")
                    print("-" * 25)
                    # Truncate very long prompts
                    if len(prompt) > 800:
                        print(prompt[:400] + "\n... [TRUNCATED] ...\n" + prompt[-400:])
                    else:
                        print(prompt)
                    print()
                
                # Display response
                if 'response' in last_call:
                    response = last_call['response']
                    print(f"🤖 LLM RESPONSE:")
                    print("-" * 15)
                    if isinstance(response, dict):
                        for key, value in response.items():
                            print(f"  {key}: {value}")
                    else:
                        print(f"  {response}")
                    print()
                
                # Display metadata
                if 'metadata' in last_call:
                    metadata = last_call['metadata']
                    print(f"📊 METADATA:")
                    print("-" * 10)
                    for key, value in metadata.items():
                        if key not in ['prompt', 'response']:  # Avoid duplication
                            print(f"  {key}: {value}")
                    print()
                
                # Display timing if available
                if 'timestamp' in last_call:
                    print(f"⏱️  Timestamp: {last_call['timestamp']}")
                    
                # Display token usage if available
                if 'usage' in last_call:
                    usage = last_call['usage']
                    print(f"📈 Token Usage:")
                    for key, value in usage.items():
                        print(f"  {key}: {value}")
                        
            else:
                print("Last call data not available")
                
        except Exception as e:
            print(f"Error retrieving DSPy history: {e}")
            print("Note: Make sure you're using a recent version of DSPy with inspect_history support")
            
        print("=" * 80)

    def display_hint_info(self, obs: Dict[str, Any]):
        """Display LLM hint information."""
        hint_value = obs.get('hint', 7)  # Default to "none"
        hint_available = obs.get('hint_available', 0)
        
        print("\n" + "🤖 LLM HINT INFORMATION".center(60, "="))
        print(f"Hint Available: {'✅ YES' if hint_available else '❌ NO'}")
        
        if hint_available:
            if self.hint_type == "subgoal":
                hint_name = SUBGOAL_NAMES.get(hint_value, f"unknown_{hint_value}")
                print(f"Suggested Subgoal: {hint_name} (value: {hint_value})")
                
                # Provide explanation of subgoal
                explanations = {
                    "CloseSubgoal": "🚪 Close a door",
                    "OpenSubgoal": "🔓 Open a door", 
                    "DropSubgoal": "📤 Drop an object",
                    "PickupSubgoal": "📥 Pick up an object",
                    "GoNextToSubgoal": "🎯 Move next to a target object",
                    "ExploreSubgoal": "🔍 Explore the environment",
                    "done": "✅ Task completed",
                    "none": "❓ No specific subgoal"
                }
                explanation = explanations.get(hint_name, "Unknown subgoal")
                print(f"Explanation: {explanation}")
                
            else:  # action
                action_name = ACTION_NAMES.get(hint_value, f"unknown_{hint_value}")
                print(f"Suggested Action: {action_name} (value: {hint_value})")
                
                # Provide explanation of action
                explanations = {
                    "turn_left": "↪️ Turn left (counterclockwise)",
                    "turn_right": "↩️ Turn right (clockwise)",
                    "move_forward": "⬆️ Move forward one step",
                    "pickup": "📥 Pick up object in front",
                    "drop": "📤 Drop carried object",
                    "toggle": "🔄 Open/close door or use object",
                    "done": "✅ End episode (task complete)"
                }
                explanation = explanations.get(action_name, "Unknown action")
                print(f"Explanation: {explanation}")
        else:
            print("LLM hint not available (may be disabled or failed)")
        
        print("=" * 60)

    def get_user_action(self) -> Optional[int]:
        """Get action input from user."""
        print("\n" + "🎮 YOUR TURN".center(50, "-"))
        print("Available actions:")
        print("  0: turn_left     1: turn_right    2: move_forward")
        print("  3: pickup        4: drop          5: toggle")
        print("  6: done")
        print("")
        print("Special commands:")
        print("  r: reset episode    q: quit           s: save screenshot")
        llm_status = "ON" if self.show_llm_details else "OFF"
        print(f"  d: toggle LLM details ({llm_status})")
        
        while True:
            try:
                user_input = input("Enter action (0-6, r, q, s, d): ").strip().lower()
                
                if user_input == 'q':
                    return 'quit'
                elif user_input == 'r':
                    return 'reset'
                elif user_input == 's':
                    return 'save'
                elif user_input == 'd':
                    return 'toggle_details'
                else:
                    action = int(user_input)
                    if 0 <= action <= 6:
                        return action
                    else:
                        print("❌ Action must be between 0-6")
                        
            except ValueError:
                print("❌ Invalid input. Enter a number 0-6, 'r', 'q', 's', or 'd'")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return 'quit'

    def save_screenshot(self, obs: Dict[str, Any]):
        """Save current state as screenshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        filename = f"llm_test_step_{self.step_count}_{timestamp}.png"
        self.display_observation(obs, save_path=filename)
        return filename

    def log_interaction(self, obs: Dict[str, Any], action: int, reward: float, done: bool):
        """Log the current interaction for later analysis."""
        base_obs = {k: v for k, v in obs.items() if k not in ['hint', 'hint_available']}
        
        # Convert NumPy types to Python native types for JSON serialization
        agent_pos = base_obs.get('agent_pos')
        if agent_pos is not None:
            agent_pos = [int(x) for x in agent_pos]
        
        log_entry = {
            'step': self.step_count,
            'episode': self.episode_count,
            'timestamp': datetime.now().isoformat(),
            'mission': base_obs.get('mission', ''),
            'agent_pos': agent_pos,
            'agent_direction': int(base_obs.get('direction', 0)) if base_obs.get('direction') is not None else None,
            'action_taken': int(action),
            'action_name': ACTION_NAMES.get(action, f"unknown_{action}"),
            'reward': float(reward),
            'done': bool(done),
            'hint_available': int(obs.get('hint_available', 0)),
            'hint_value': int(obs.get('hint', 7)),
            'hint_type': self.hint_type,
            'encoding_used': self.encoding_type,
            'encoded_state': self.encoder.encode_all(base_obs)[self.encoding_type]
        }
        
        if self.hint_type == "subgoal":
            log_entry['hint_name'] = SUBGOAL_NAMES.get(obs.get('hint', 7), 'unknown')
        else:
            log_entry['hint_name'] = ACTION_NAMES.get(obs.get('hint', 6), 'unknown')
        
        # Add LLM call details if using DSPy
        if self.hint_source == "dspy":
            try:
                import dspy
                history = dspy.inspect_history(1)
                if history:
                    last_call = history[-1]
                    log_entry['llm_call'] = {
                        'prompt': last_call.get('prompt', ''),
                        'response': last_call.get('response', ''),
                        'metadata': last_call.get('metadata', {}),
                        'timestamp': last_call.get('timestamp', ''),
                        'usage': last_call.get('usage', {})
                    }
            except Exception as e:
                log_entry['llm_call_error'] = str(e)
        
        self.session_log.append(log_entry)

    def save_session_log(self):
        """Save the complete session log."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_interactive_session_{self.env_id.replace('-', '_')}_{timestamp}.json"
        
        session_data = {
            'environment': self.env_id,
            'hint_type': self.hint_type,
            'hint_source': self.hint_source,
            'encoding_type': self.encoding_type,
            'seed': int(self.seed) if self.seed is not None else None,
            'total_steps': int(self.step_count),
            'total_episodes': int(self.episode_count),
            'session_start': self.session_log[0]['timestamp'] if self.session_log else None,
            'session_end': datetime.now().isoformat(),
            'interactions': self.session_log
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"💾 Session log saved to: {filename}")
        return filename

    def run_interactive_session(self):
        """Run the main interactive session."""
        print("\n🎯 Starting Interactive LLM Testing Session")
        print("=" * 60)
        print("You can manually control the agent while seeing LLM suggestions.")
        print("This is perfect for analyzing the LLM's reasoning!")
        print("=" * 60)
        
        try:
            # Reset environment
            obs, info = self.env.reset(seed=self.seed)
            self.step_count = 0
            self.episode_count = 0
            
            while True:
                # Display current state
                print(f"\n📍 STEP {self.step_count} | EPISODE {self.episode_count + 1}")
                print(f"Mission: {obs.get('mission', 'Unknown')}")
                
                # Show environment and encoding
                self.display_observation(obs)
                
                # Show LLM configuration details (if enabled)
                if self.show_llm_details:
                    self.display_llm_details()
                
                # Show LLM hint
                self.display_hint_info(obs)
                
                # Show DSPy call history if using DSPy (if enabled)
                if self.hint_source == "dspy" and self.show_llm_details:
                    self.display_dspy_history()
                
                # Get user action
                action = self.get_user_action()
                
                # Handle special commands
                if action == 'quit':
                    break
                elif action == 'reset':
                    print("🔄 Resetting episode...")
                    obs, info = self.env.reset()
                    self.step_count = 0
                    self.episode_count += 1
                    continue
                elif action == 'save':
                    filename = self.save_screenshot(obs)
                    print(f"📸 Screenshot saved as: {filename}")
                    continue
                elif action == 'toggle_details':
                    self.show_llm_details = not self.show_llm_details
                    status = "enabled" if self.show_llm_details else "disabled"
                    print(f"🔧 LLM details display {status}")
                    continue
                
                # Take the action
                print(f"🎬 Taking action: {ACTION_NAMES.get(action, action)}")
                next_obs, reward, done, truncated, info = self.env.step(action)
                
                # Log the interaction
                self.log_interaction(obs, action, reward, done or truncated)
                
                # Show results
                print(f"Reward: {reward}")
                if done:
                    print("🏆 Episode completed!")
                elif truncated:
                    print("⏱️  Episode truncated!")
                
                # Update state
                obs = next_obs
                self.step_count += 1
                
                # Handle episode end
                if done or truncated:
                    print(f"\n📊 Episode {self.episode_count + 1} Summary:")
                    print(f"   Steps taken: {self.step_count}")
                    print(f"   Final reward: {reward}")
                    
                    cont = input("\nStart new episode? (y/n): ").strip().lower()
                    if cont != 'y':
                        break
                    
                    # Reset for new episode
                    obs, info = self.env.reset()
                    self.step_count = 0
                    self.episode_count += 1
        
        except KeyboardInterrupt:
            print("\n👋 Session interrupted by user")
        
        finally:
            # Clean up
            plt.close('all')
            self.env.close()
            
            # Save session log
            if self.session_log:
                self.save_session_log()
                print(f"\n📈 Session completed! Total interactions: {len(self.session_log)}")
            
            print("Thanks for testing! 🎉")

def main():
    parser = argparse.ArgumentParser(description="Interactive LLM Testing for MinGrid")
    parser.add_argument("--env", type=str, default="BabyAI-GoToObj-v0",
                       help="Environment ID (default: BabyAI-GoToObj-v0)")
    parser.add_argument("--hint-type", type=str, choices=["subgoal", "action"], 
                       default="subgoal", help="Type of hints (default: subgoal)")
    parser.add_argument("--hint-source", type=str, choices=["dspy", "babyai_bot"],
                       default="dspy", help="Source of hints (default: dspy)")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to fine-tuned model (optional)")
    parser.add_argument("--encoding", type=str, 
                       choices=["natural", "ascii", "tuples", "relative"],
                       default="ascii", help="Text encoding type (default: ascii)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (optional)")
    parser.add_argument("--hide-llm-details", action="store_true",
                       help="Hide LLM configuration and call details by default")
    
    args = parser.parse_args()
    
    # Create and run the interactive tester
    tester = InteractiveLLMTester(
        env_id=args.env,
        hint_type=args.hint_type,
        hint_source=args.hint_source,
        hint_model_path=args.model_path,
        encoding_type=args.encoding,
        seed=args.seed,
        show_llm_details=not args.hide_llm_details
    )
    
    tester.run_interactive_session()

if __name__ == "__main__":
    main()
