#!/usr/bin/env python3
"""
LLM Evaluation Script
Evaluates DSPy models on unified test datasets.
"""

import os
import sys
import json
import random
from typing import Dict, List, Any, Optional
from collections import Counter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
from utils.dspy_signature import NextStepSignature
from utils.observation_encoder import ObservationEncoder

def load_test_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load test dataset from unified dataset directory."""
    test_path = os.path.join(dataset_path, "test.json")
    with open(test_path, 'r') as f:
        return json.load(f)

def sample_balanced_test_data(demonstrations: List[Dict], hint_type: str, 
                             testset_size: int = 10) -> List[Dict]:
    """Sample balanced test data with equal proportions from each category."""
    # Set random seed for reproducible sampling
    random.seed(42)
    
    # Extract all samples with their categories
    all_samples = []
    category_counts = Counter()
    
    for demo in demonstrations:
        for episode in demo["episodes"]:
            for step_data in episode["observations"]:
                if hint_type == "subgoal":
                    category = step_data.get("subgoal", "unknown")
                else:  # action
                    category = str(step_data.get("action", "unknown"))
                
                all_samples.append({
                    "step_data": step_data,
                    "category": category,
                    "demo": demo
                })
                category_counts[category] += 1
    
    print(f"Available test categories: {dict(category_counts)}")
    
    # Group samples by category for equal sampling
    samples_by_category = {}
    for sample in all_samples:
        category = sample["category"]
        if category not in samples_by_category:
            samples_by_category[category] = []
        samples_by_category[category].append(sample)
    
    # Randomize samples within each category
    for category in samples_by_category:
        random.shuffle(samples_by_category[category])
    
    # Calculate equal samples per category
    available_categories = list(samples_by_category.keys())
    num_categories = len(available_categories)
    
    if num_categories == 0:
        print("No categories found!")
        return []
    
    # Equal distribution: divide testset_size by number of categories
    samples_per_category = testset_size // num_categories
    remaining_samples = testset_size % num_categories
    
    print(f"Sampling strategy: {samples_per_category} samples per category, {remaining_samples} extra samples")
    
    balanced_samples = []
    total_samples = 0
    
    # Sample equal amounts from each category
    for i, category in enumerate(available_categories):
        category_samples = samples_by_category[category]
        
        # Base samples per category
        num_to_sample = samples_per_category
        
        # Distribute remaining samples to first few categories
        if i < remaining_samples:
            num_to_sample += 1
        
        # Don't sample more than available
        num_to_sample = min(num_to_sample, len(category_samples))
        
        if num_to_sample > 0:
            sampled = category_samples[:num_to_sample]  # Take first N after shuffle
            balanced_samples.extend(sampled)
            total_samples += len(sampled)
            print(f"Category '{category}': sampled {len(sampled)}/{len(category_samples)} samples")
    
    # Final shuffle of all selected samples for randomness
    random.shuffle(balanced_samples)
    
    print(f"Total balanced samples: {len(balanced_samples)}")
    
    return balanced_samples[:testset_size]

def create_test_examples(samples: List[Dict], encoding_type: str, 
                        hint_type: str) -> List[Dict]:
    """Create test examples from samples."""
    test_examples = []
    
    for sample in samples:
        step_data = sample["step_data"]
        encoded_obs = step_data["encoded_observations"]
        
        # Get the specific encoding type
        if encoding_type == "natural":
            observation_text = encoded_obs.get("natural", "")
        elif encoding_type == "ascii":
            observation_text = encoded_obs.get("ascii", "")
        elif encoding_type == "tuples":
            observation_text = encoded_obs.get("tuples", "")
        elif encoding_type == "relative":
            observation_text = encoded_obs.get("relative", "")
        else:
            observation_text = encoded_obs.get("ascii", "")
        
        # Create target based on hint type
        if hint_type == "subgoal":
            target = step_data.get("subgoal", "unknown")
        else:  # action
            target = str(step_data.get("action", "unknown"))
        
        test_examples.append({
            "observation": observation_text,
            "target": target,
            "step_data": step_data
        })
    
    return test_examples

class Evaluate:
    """Evaluates DSPy models on test datasets."""
    
    def __init__(self, dataset_path: str, hint_type: str = "subgoal", 
                 encoding_type: str = "ascii", model_path: Optional[str] = None,
                 testset_size: int = 10):
        self.dataset_path = dataset_path
        self.hint_type = hint_type
        self.encoding_type = encoding_type
        self.model_path = model_path
        self.testset_size = testset_size
        
        # Configure LLM
        from utils.dspy_signature import configure_llm
        configure_llm()  # Uses default source='GROQ'
        
        # Load or create predictor
        if model_path and os.path.exists(model_path):
            print(f"Loading saved model from: {model_path}")
            self.predictor = dspy.Predict(NextStepSignature)
            self.predictor.load(model_path)
        else:
            print("Using direct LLM (no saved model)")
            self.predictor = dspy.Predict(NextStepSignature)
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on the test dataset."""
        print(f"Evaluating {self.hint_type} predictor with {self.encoding_type} encoding")
        print(f"Dataset: {self.dataset_path}")
        print(f"Test set size: {self.testset_size}")
        
        # Load test dataset
        demonstrations = load_test_dataset(self.dataset_path)
        print(f"Loaded {len(demonstrations)} test demonstrations")
        
        # Sample balanced test data
        balanced_samples = sample_balanced_test_data(demonstrations, self.hint_type, self.testset_size)
        print(f"Sampled {len(balanced_samples)} balanced test samples")
        
        # Create test examples
        test_examples = create_test_examples(balanced_samples, self.encoding_type, self.hint_type)
        print(f"Created {len(test_examples)} test examples")
        
        # Run evaluation
        correct_predictions = 0
        total_predictions = 0
        predictions = []
        
        for i, example in enumerate(test_examples):
            try:
                # Get task description
                from utils.config_task_desc import task_desc
                
                # Extract current state from the example
                current_state = example["observation"]  # This is the encoded observation
                
                # For evaluation, we don't have previous actions context, so use empty string
                previous_actions = ""
                
                # Make prediction with proper field mapping
                prediction = self.predictor(
                    task_description=task_desc,
                    current_state=current_state,
                    previous_actions=previous_actions
                )
                
                # Extract predicted value (access as attribute, not dict)
                if self.hint_type == "subgoal":
                    predicted = getattr(prediction, "subgoal", "unknown")
                else:  # action
                    predicted = getattr(prediction, "primitive_action", "unknown")
                
                target = example["target"]
                
                # Check if prediction is correct
                is_correct = str(predicted).strip().lower() == str(target).strip().lower()
                if is_correct:
                    correct_predictions += 1
                
                total_predictions += 1
                
                predictions.append({
                    "example_id": i,
                    "observation": example["observation"][:100] + "..." if len(example["observation"]) > 100 else example["observation"],
                    "target": target,
                    "predicted": predicted,
                    "correct": is_correct
                })
                
                print(f"Example {i+1}/{len(test_examples)}: {'✓' if is_correct else '✗'} "
                      f"Target: {target}, Predicted: {predicted}")
                
            except Exception as e:
                print(f"Error evaluating example {i}: {e}")
                predictions.append({
                    "example_id": i,
                    "observation": example["observation"][:100] + "..." if len(example["observation"]) > 100 else example["observation"],
                    "target": example["target"],
                    "predicted": "ERROR",
                    "correct": False,
                    "error": str(e)
                })
                total_predictions += 1
        
        # Calculate accuracy
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Generate results
        results = {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "testset_size": self.testset_size,
            "hint_type": self.hint_type,
            "encoding_type": self.encoding_type,
            "model_path": self.model_path,
            "dataset_path": self.dataset_path,
            "predictions": predictions
        }
        
        print(f"\n📊 EVALUATION RESULTS:")
        print(f"Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
        print(f"Hint type: {self.hint_type}")
        print(f"Encoding type: {self.encoding_type}")
        print(f"Model: {'Saved' if self.model_path else 'Direct LLM'}")
        
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate DSPy models")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--hint-type", type=str, default="subgoal",
                       choices=["subgoal", "action"],
                       help="Type of hint to predict")
    parser.add_argument("--encoding-type", type=str, default="ascii",
                       choices=["ascii", "natural", "tuples", "relative"],
                       help="Text encoding type")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to saved DSPy model (if None, uses direct LLM)")
    parser.add_argument("--testset-size", type=int, default=10,
                       help="Number of test samples to evaluate on")
    
    args = parser.parse_args()
    
    evaluator = Evaluate(
        dataset_path=args.dataset_path,
        hint_type=args.hint_type,
        encoding_type=args.encoding_type,
        model_path=args.model_path,
        testset_size=args.testset_size
    )
    
    results = evaluator.evaluate()
    print(f"Evaluation completed with {results['accuracy']:.3f} accuracy")
