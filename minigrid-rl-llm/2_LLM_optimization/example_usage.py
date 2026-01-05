#!/usr/bin/env python3
"""
Example usage of DSPy Agent Fine-tuning
Demonstrates the complete workflow from fine-tuning to loading and using the model.
"""

import os
import sys
from dspy_agent_fine_tuning import run_dspy_agent_fine_tuning, MinigridAgent

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
from utils.dspy_signature import configure_llm

def example_fine_tuning():
    """Example of running the fine-tuning pipeline."""
    print("🚀 Example: DSPy Agent Fine-tuning")
    print("=" * 50)
    
    # Define paths and parameters
    dataset_path = "../1_GT_collection/GT_dataset/dataset_9env_5seed_5episodes_v2_0623_2116"
    output_dir = "./example_models"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please run the GT collection script first or provide a valid dataset path.")
        return None
    
    print(f"📁 Using dataset: {dataset_path}")
    print(f"💾 Models will be saved to: {output_dir}")
    
    # Run fine-tuning with example parameters
    try:
        model_path = run_dspy_agent_fine_tuning(
            dataset_path=dataset_path,
            teacher_model="llama-3.3-70b-versatile",  # Large model via GROQ
            student_model="llama3.1:70b",             # Local model via Ollama
            teacher_groq=True,                        # Use GROQ for teacher
            student_groq=False,                       # Use Ollama for student
            hint_type="action",                       # Predict actions
            encoding_type="ascii",                    # Use ASCII encoding
            samples_per_category=5,                   # Small sample size for example
            max_bootstrapped_demos=1,                 # Light optimization
            minibatch_size=20,                        # Small batches
            assessor_model="deepseek-r1-distill-llama-70b",  # AI feedback assessor
            assessor_groq=True,                       # Use GROQ for assessor
            output_dir=output_dir
        )
        
        if model_path:
            print(f"\n✅ Fine-tuning completed successfully!")
            print(f"📦 Model saved at: {model_path}")
            return model_path
        else:
            print(f"\n⚠️ Fine-tuning completed but model save failed.")
            return None
            
    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        return None

def example_load_and_use_model(model_path: str):
    """Example of loading and using a fine-tuned model."""
    print(f"\n🔧 Example: Loading and Using Fine-tuned Model")
    print("=" * 50)
    
    try:
        # Configure the same model that was used for fine-tuning
        print("🔧 Configuring LLM...")
        configure_llm(source="local")  # Automatically uses llama3.1:latest for local
        
        # Create agent and load the fine-tuned model
        print(f"📦 Loading model from: {model_path}")
        agent = MinigridAgent()
        agent.load(model_path)
        
        print("✅ Model loaded successfully!")
        
        # Example observations to test the model
        test_observations = [
            "Grid: 7x7. Agent at (1,1) facing east. Red door at (3,3). Key at (5,2). Task: reach the red door.",
            "Grid: 5x5. Agent at (2,2) facing north. Green ball at (4,4). Open door at (3,1). Task: pick up green ball.",
            "Grid: 6x6. Agent at (0,0) facing south. Wall at (2,2). Box at (4,4). Task: move box to (1,1)."
        ]
        
        print(f"\n🧪 Testing model with {len(test_observations)} observations...")
        
        for i, observation in enumerate(test_observations, 1):
            try:
                print(f"\n--- Test {i} ---")
                print(f"Observation: {observation[:80]}...")
                
                # Get prediction from the fine-tuned model
                prediction = agent(observation=observation)
                
                # Action mapping for readability
                action_map = {
                    0: "Turn left",
                    1: "Turn right", 
                    2: "Move forward",
                    3: "Pick up",
                    4: "Drop",
                    5: "Toggle",
                    6: "Done"
                }
                
                action_id = prediction.primitive_action
                action_name = action_map.get(int(action_id), f"Unknown ({action_id})")
                
                print(f"Predicted Action: {action_id} ({action_name})")
                
            except Exception as e:
                print(f"❌ Error testing observation {i}: {e}")
        
        print(f"\n✅ Model testing completed!")
        
    except Exception as e:
        print(f"❌ Error loading or using model: {e}")

def example_integration_with_rl():
    """Example of how to integrate with RL training."""
    print(f"\n🤖 Example: Integration with RL Training")
    print("=" * 50)
    
    print("The fine-tuned model can be integrated with RL training in several ways:")
    print()
    print("1. **As a Hint Provider**:")
    print("   - Use with HintWrapper in RL training")
    print("   - Provide action suggestions to guide RL agent")
    print()
    print("2. **As Curriculum Learning**:")
    print("   - Start RL training with hints from fine-tuned model")
    print("   - Gradually reduce hint frequency as agent improves")
    print()
    print("3. **As Evaluation Baseline**:")
    print("   - Compare RL agent performance against fine-tuned LLM")
    print("   - Use for ablation studies")
    print()
    
    example_code = '''
# Example: Using fine-tuned model with RL training
from utils.hint_wrapper import HintWrapper
import gymnasium as gym

# Create environment with fine-tuned model as hint source
env = gym.make("BabyAI-GoToObj-v0")
env = HintWrapper(
    env, 
    hint_source="dspy",
    hint_model_path="./example_models/finetuned_model.pkl",
    hint_type="action",
    hint_frequency=5,  # Provide hints every 5 steps
    encoding_type="ascii"
)

# Now use this environment for RL training
# The RL agent will receive hints from the fine-tuned LLM
'''
    
    print("Example Integration Code:")
    print(example_code)

def main():
    """Main example workflow."""
    print("🎯 DSPy Agent Fine-tuning - Complete Example")
    print("=" * 60)
    print("This example demonstrates:")
    print("1. Fine-tuning a DSPy agent using teacher-student approach")
    print("2. Loading and testing the fine-tuned model")
    print("3. Integration patterns with RL training")
    print("=" * 60)
    
    # Step 1: Run fine-tuning
    model_path = example_fine_tuning()
    
    if model_path and os.path.exists(model_path):
        # Step 2: Load and test the model
        example_load_and_use_model(model_path)
    else:
        print("\n⚠️ Skipping model testing due to fine-tuning issues.")
    
    # Step 3: Show integration examples
    example_integration_with_rl()
    
    print(f"\n🎉 Example workflow completed!")
    print("=" * 60)
    print("Next steps:")
    print("- Try different teacher/student model combinations")
    print("- Experiment with different encoding types")
    print("- Use the fine-tuned model in RL training")
    print("- Evaluate performance on held-out test sets")

if __name__ == "__main__":
    main() 