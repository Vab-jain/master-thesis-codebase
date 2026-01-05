#!/usr/bin/env python3
"""
DSPy Agent Fine-tuning Script

Implements the DSPy teacher-student fine-tuning approach with AI feedback metrics:
1. Optimize prompts using a larger teacher model (GROQ API)
2. Fine-tune a smaller student model using knowledge distillation (local/GROQ)
3. Enhanced evaluation with AI feedback for nuanced action assessment
4. Save fine-tuned models locally for deployment

Based on: https://dspy.ai/tutorials/games/
AI Feedback Metrics: https://dspy.ai/learn/evaluation/metrics/
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import Counter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
from utils.dspy_signature import NextStepSignature, configure_llm
from utils.observation_encoder import ObservationEncoder

class MinigridAgent(dspy.Module):
    """DSPy Agent for Minigrid navigation tasks."""
    
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(NextStepSignature)
    
    def forward(self, observation):
        """Forward pass through the agent."""
        # For simplicity, we'll use the observation as both task description and current state
        # In practice, you might want to separate these
        prediction = self.predict(
            task_description="Navigate and complete the given task",
            current_state=observation,
            previous_actions=0  # Simplified for this example
        )
        return prediction

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load training dataset from unified dataset directory."""
    train_path = os.path.join(dataset_path, "train.json")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training dataset not found at: {train_path}")
    
    try:
        with open(train_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"Invalid dataset format: expected non-empty list, got {type(data)}")
        
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in dataset file {train_path}: {e}")

def sample_balanced_data(demonstrations: List[Dict], hint_type: str, 
                        samples_per_category: int = 10) -> List[Dict]:
    """Sample balanced data with similar number of samples per category."""
    if not demonstrations:
        raise ValueError("No demonstrations provided for sampling")
    
    all_samples = []
    category_counts = Counter()
    
    # Extract all samples with their categories
    for demo in demonstrations:
        if "episodes" not in demo:
            print(f"⚠️ Skipping demo without episodes: {demo.keys()}")
            continue
            
        for episode in demo["episodes"]:
            if "observations" not in episode:
                print(f"⚠️ Skipping episode without observations")
                continue
                
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
    
    if not all_samples:
        raise ValueError("No valid samples found in demonstrations")
    
    print(f"📊 Available categories: {dict(category_counts)}")
    
    # Sample balanced data
    balanced_samples = []
    for category, count in category_counts.items():
        category_samples = [s for s in all_samples if s["category"] == category]
        num_to_sample = min(samples_per_category, count)
        if num_to_sample > 0:
            sampled = random.sample(category_samples, num_to_sample)
            balanced_samples.extend(sampled)
            print(f"  Category '{category}': sampled {num_to_sample}/{count} samples")
    
    return balanced_samples

def create_training_examples(samples: List[Dict], encoding_type: str, 
                           hint_type: str) -> List[dspy.Example]:
    """Create DSPy training examples from samples."""
    if not samples:
        raise ValueError("No samples provided for creating training examples")
    
    examples = []
    skipped_count = 0

    for sample in samples:
        try:
            step_data = sample["step_data"]
            
            # Validate step data structure
            if "encoded_observations" not in step_data:
                skipped_count += 1
                continue
                
            encoded_obs = step_data["encoded_observations"]
            
            # Get the specific encoding type
            observation_text = encoded_obs.get(encoding_type, "")
            
            # Fallback to ascii if requested encoding is empty
            if not observation_text.strip() and encoding_type != "ascii":
                observation_text = encoded_obs.get("ascii", "")
            
            # Skip if no valid observation text found
            if not observation_text.strip():
                skipped_count += 1
                continue
            
            # Create target based on hint type
            if hint_type == "subgoal":
                target = step_data.get("subgoal", "unknown")
            else:  # action
                target = str(step_data.get("action", "unknown"))
            
            # Skip if target is missing or invalid
            if target == "unknown" or target is None:
                skipped_count += 1
                continue
            
            # Create DSPy example
            example = dspy.Example(
                observation=observation_text,
                primitive_action=target if hint_type == "action" else target
            ).with_inputs("observation")
            
            examples.append(example)
            
        except Exception as e:
            print(f"⚠️ Error processing sample: {e}")
            skipped_count += 1
            continue
    
    if skipped_count > 0:
        print(f"⚠️ Skipped {skipped_count} invalid samples out of {len(samples)} total")
    
    if not examples:
        raise ValueError("No valid training examples could be created from samples")
    
    return examples

class ActionAssessment(dspy.Signature):
    """Assess whether a predicted action makes sense given the current state and reasoning."""
    
    current_state: str = dspy.InputField(desc="Current environment state description")
    predicted_action: int = dspy.InputField(desc="Action predicted by the agent (0-6)")
    explanation: str = dspy.InputField(desc="Agent's reasoning/explanation for the action")
    ground_truth_action: int = dspy.InputField(desc="Expected correct action")
    
    assessment_question: str = dspy.InputField(desc="Question asking if the action makes sense")
    makes_sense: bool = dspy.OutputField(desc="Whether the predicted action and reasoning make sense in context")

class ActionMetricAgent(dspy.Module):
    """DSPy module that evaluates action predictions using AI feedback."""
    
    def __init__(self, assessor_model="deepseek-r1-distill-llama-70b", assessor_groq=True):
        super().__init__()
        
        try:
            # Configure the assessor LLM
            if assessor_groq:
                # Import GROQ API key from utils
                import utils
                assessor_lm = dspy.LM(
                    assessor_model, 
                    api_base='https://api.groq.com/openai/v1',
                    api_key=utils.GROQ_API_KEY
                )
            else:
                assessor_lm = dspy.LM(assessor_model, api_base='http://localhost:11434')
            
            # Create the assessment predictor with the specific LLM
            self.assessor = dspy.Predict(ActionAssessment)
            self.assessor.lm = assessor_lm
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize assessor LLM {assessor_model}: {e}")
            print("AI feedback assessment will be disabled.")
            self.assessor = None
    
    def forward(self, current_state, predicted_action, explanation, ground_truth_action):
        """Evaluate if the predicted action makes sense given the context."""
        
        # Check if assessor is available
        if self.assessor is None:
            print("⚠️ Assessor not available, returning False for AI assessment")
            return False
        
        # Create assessment question
        action_names = {
            0: "turn left", 1: "turn right", 2: "move forward", 
            3: "pick up", 4: "drop", 5: "toggle", 6: "done"
        }
        
        pred_action_name = action_names.get(int(predicted_action), f"action {predicted_action}")
        gt_action_name = action_names.get(int(ground_truth_action), f"action {ground_truth_action}")
        
        assessment_question = (
            f"The agent predicted '{pred_action_name}' (action {predicted_action}) instead of "
            f"the expected '{gt_action_name}' (action {ground_truth_action}). "
            f"Given the current state and the agent's reasoning, does the predicted action "
            f"make logical sense as a reasonable alternative, even if not optimal?"
        )
        
        # Get assessment from the third LLM
        try:
            assessment = self.assessor(
                current_state=current_state,
                predicted_action=predicted_action,
                explanation=explanation,
                ground_truth_action=ground_truth_action,
                assessment_question=assessment_question
            )
            
            return assessment.makes_sense
            
        except Exception as e:
            print(f"⚠️ Assessment failed: {e}")
            return False

# Global assessor instance (initialized lazily)
_metric_assessor = None

def get_metric_assessor():
    """Get or create the global metric assessor."""
    global _metric_assessor
    if _metric_assessor is None:
        _metric_assessor = ActionMetricAgent()
    return _metric_assessor

def accuracy_metric(example, pred, trace=None):
    """
    Enhanced accuracy metric with AI feedback for evaluation.
    
    - If predicted action matches ground truth: return 1
    - If different: ask third LLM if prediction makes sense given context
    - Returns float score for evaluation, bool for optimization
    """
    
    # Check if we have the required attributes
    if not (hasattr(example, 'primitive_action') and hasattr(pred, 'primitive_action')):
        return False if trace is not None else 0.0
    
    # Extract values
    try:
        ground_truth = int(example.primitive_action)
        predicted = int(pred.primitive_action)
    except (ValueError, TypeError):
        return False if trace is not None else 0.0
    
    # Exact match case - perfect score
    if predicted == ground_truth:
        return True if trace is not None else 1.0
    
    # Different prediction - use AI feedback assessment
    try:
        # Get current state from example (try different possible field names)
        current_state = ""
        if hasattr(example, 'observation'):
            current_state = str(example.observation)
        elif hasattr(example, 'current_state'):
            current_state = str(example.current_state)
        elif hasattr(example, 'state'):
            current_state = str(example.state)
        
        # Get explanation from prediction (if available)
        explanation = ""
        if hasattr(pred, 'rationale'):
            explanation = str(pred.rationale)
        elif hasattr(pred, 'reasoning'):
            explanation = str(pred.reasoning)
        elif hasattr(pred, 'explanation'):
            explanation = str(pred.explanation)
        else:
            explanation = "No explanation provided by the agent."
        
        # Skip AI assessment if we don't have enough context
        if not current_state.strip():
            return False if trace is not None else 0.0
        
        # Get the assessor and evaluate
        assessor = get_metric_assessor()
        makes_sense = assessor(
            current_state=current_state,
            predicted_action=predicted,
            explanation=explanation,
            ground_truth_action=ground_truth
        )
        
        # Return appropriate score format
        if trace is not None:
            # During optimization: return bool (True only if makes sense)
            return bool(makes_sense)
        else:
            # During evaluation: return float (0.5 if makes sense, 0.0 if not)
            return 0.5 if makes_sense else 0.0
            
    except Exception as e:
        print(f"⚠️ AI feedback assessment failed: {e}")
        # Fallback to simple mismatch
        return False if trace is not None else 0.0

def run_dspy_agent_fine_tuning(
    dataset_path: str,
    teacher_model: str = "llama-3.3-70b-versatile",
    student_model: str = "llama3.1:70b",
    teacher_groq: bool = True,
    student_groq: bool = False,
    hint_type: str = "action",
    encoding_type: str = "ascii",
    samples_per_category: int = 10,
    max_bootstrapped_demos: int = 1,
    minibatch_size: int = 40,
    assessor_model: str = "deepseek-r1-distill-llama-70b",
    assessor_groq: bool = True,
    output_dir: str = "saved_llm_models"
) -> str:
    """
    Run the complete DSPy agent fine-tuning pipeline.
    
    Args:
        dataset_path: Path to the dataset directory
        teacher_model: Model ID for the teacher (larger model)
        student_model: Model ID for the student (smaller model to fine-tune)
        teacher_groq: Whether teacher model uses GROQ API
        student_groq: Whether student model uses GROQ API  
        hint_type: Type of hint to predict ("action" or "subgoal")
        encoding_type: Text encoding type
        samples_per_category: Number of samples per category
        max_bootstrapped_demos: Max bootstrapped demonstrations
        minibatch_size: Minibatch size for optimization
        assessor_model: Model ID for the metric assessor (third LLM)
        assessor_groq: Whether assessor model uses GROQ API
        output_dir: Directory to save the fine-tuned model
    
    Returns:
        Path to the saved fine-tuned model
    """
    
    print("🎯 Starting DSPy Agent Fine-tuning Pipeline")
    print("=" * 60)
    print(f"📁 Dataset: {dataset_path}")
    print(f"🧠 Teacher Model: {teacher_model} (GROQ: {teacher_groq})")
    print(f"🎓 Student Model: {student_model} (GROQ: {student_groq})")
    print(f"📊 Assessor Model: {assessor_model} (GROQ: {assessor_groq})")
    print(f"🎯 Hint Type: {hint_type} | 📝 Encoding: {encoding_type}")
    print("=" * 60)
    
    try:
        # Initialize the global metric assessor with specified parameters
        global _metric_assessor
        _metric_assessor = ActionMetricAgent(assessor_model=assessor_model, assessor_groq=assessor_groq)
        print(f"🤖 AI Feedback Metric initialized with {assessor_model}")
        
        # Load and prepare dataset
        print("\n📊 Loading and preparing dataset...")
        demonstrations = load_dataset(dataset_path)
        print(f"✅ Loaded {len(demonstrations)} demonstrations")
        
        # Sample balanced data
        balanced_samples = sample_balanced_data(demonstrations, hint_type, samples_per_category)
        print(f"✅ Sampled {len(balanced_samples)} balanced samples")
        
        # Create training examples
        training_examples = create_training_examples(balanced_samples, encoding_type, hint_type)
        print(f"✅ Created {len(training_examples)} training examples")
        
    except Exception as e:
        print(f"❌ Dataset preparation failed: {e}")
        return None
    
    # Split into train and dev sets
    random.shuffle(training_examples)
    split_idx = int(0.8 * len(training_examples))
    trainset = training_examples[:split_idx]
    devset = training_examples[split_idx:]
    
    print(f"Training set: {len(trainset)} examples")
    print(f"Development set: {len(devset)} examples")
    
    # Step 1: Configure teacher model and optimize prompts
    print(f"\n🧠 Step 1: Optimizing prompts with teacher model ({teacher_model})...")
    configure_llm(llm_model_id=teacher_model, source='GROQ' if teacher_groq else 'local')
    
    # Create teacher agent
    teacher_agent = MinigridAgent()
    
    # Optimize prompts using MIPROv2
    print("🔧 Running prompt optimization with MIPROv2...")
    optimizer = dspy.MIPROv2(
        metric=accuracy_metric, 
        auto="light", 
        num_threads=4,  # Reduced for stability
        prompt_model=dspy.LM(teacher_model, api_base='https://api.groq.com/openai/v1' if teacher_groq else 'http://localhost:11434')
    )
    
    config = dict(
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=0,
        minibatch_size=min(minibatch_size, len(trainset))
    )
    
    try:
        optimized_teacher = optimizer.compile(
            teacher_agent, 
            trainset=trainset, 
            **config, 
            requires_permission_to_run=False
        )
        print("✅ Prompt optimization completed successfully")
    except Exception as e:
        print(f"⚠️ Warning: Prompt optimization failed ({e}). Using base teacher model.")
        optimized_teacher = teacher_agent
    
    # Step 2: Configure student model
    print(f"\n🎓 Step 2: Setting up student model ({student_model})...")
    configure_llm(llm_model_id=student_model, source='GROQ' if student_groq else 'local')
    
    # Create student from optimized teacher
    student_agent = optimized_teacher.deepcopy()
    student_agent.set_lm(dspy.LM(student_model, api_base='http://localhost:11434' if not student_groq else 'https://api.groq.com/openai/v1'))
    
    # Step 3: Fine-tune student model
    print(f"\n🔥 Step 3: Fine-tuning student model...")
    finetune_optimizer = dspy.BootstrapFinetune(metric=accuracy_metric, num_threads=4)
    
    try:
        finetuned_student = finetune_optimizer.compile(
            student_agent, 
            teacher=optimized_teacher, 
            trainset=trainset
        )
        print("✅ Fine-tuning completed successfully")
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        print("Using prompt-optimized student model instead")
        finetuned_student = student_agent
    
    # Step 4: Evaluate the fine-tuned model
    print(f"\n📊 Step 4: Evaluating fine-tuned model...")
    if devset:
        try:
            # Limit evaluation size to avoid quota issues
            eval_size = min(20, len(devset))
            eval_subset = devset[:eval_size]
            
            correct = 0
            exact_matches = 0
            ai_reasonable = 0
            total = 0
            
            for example in eval_subset:
                try:
                    prediction = finetuned_student(observation=example.observation)
                    
                    # Check exact match
                    exact_match = str(example.primitive_action) == str(prediction.primitive_action)
                    if exact_match:
                        exact_matches += 1
                        correct += 1
                    else:
                        # Check AI assessment
                        metric_score = accuracy_metric(example, prediction)
                        if metric_score > 0:  # 0.5 for reasonable, 0.0 for unreasonable
                            ai_reasonable += 1
                            correct += 1
                    
                    total += 1
                    
                except Exception as e:
                    print(f"⚠️ Evaluation error on example: {e}")
                    continue
            
            if total > 0:
                accuracy = correct / total
                exact_rate = exact_matches / total
                ai_rate = ai_reasonable / (total - exact_matches) if (total - exact_matches) > 0 else 0
                
                print(f"📈 Overall Accuracy: {accuracy:.3f} ({correct}/{total})")
                print(f"  🎯 Exact Matches: {exact_rate:.3f} ({exact_matches}/{total})")
                print(f"  🤖 AI-Assessed Reasonable: {ai_rate:.3f} ({ai_reasonable}/{total - exact_matches})")
            else:
                print("⚠️ No valid evaluations completed")
            
        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}")
    
    # Step 5: Save the fine-tuned model
    print(f"\n💾 Step 5: Saving fine-tuned model...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"finetuned_{student_model.replace(':', '_').replace('-', '_')}_{hint_type}_{encoding_type}_{timestamp}"
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    
    try:
        finetuned_student.save(model_path)
        print(f"✅ Fine-tuned model saved to: {model_path}")
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        # Try alternative save format
        alt_path = os.path.join(output_dir, f"{model_name}.json")
        try:
            with open(alt_path, 'w') as f:
                json.dump({
                    "model_type": "dspy_finetuned",
                    "student_model": student_model,
                    "teacher_model": teacher_model,
                    "hint_type": hint_type,
                    "encoding_type": encoding_type,
                    "timestamp": timestamp,
                    "error": str(e)
                }, f, indent=2)
            print(f"⚠️ Saved metadata only to: {alt_path}")
            model_path = alt_path
        except Exception as e2:
            print(f"❌ Failed to save metadata: {e2}")
            model_path = None
    
    # Save training info
    training_info = {
        "model_name": model_name,
        "model_path": model_path,
        "dataset_path": dataset_path,
        "teacher_model": teacher_model,
        "student_model": student_model,
        "teacher_groq": teacher_groq,
        "student_groq": student_groq,
        "assessor_model": assessor_model,
        "assessor_groq": assessor_groq,
        "hint_type": hint_type,
        "encoding_type": encoding_type,
        "samples_per_category": samples_per_category,
        "max_bootstrapped_demos": max_bootstrapped_demos,
        "minibatch_size": minibatch_size,
        "total_training_examples": len(training_examples),
        "trainset_size": len(trainset),
        "devset_size": len(devset),
        "total_demonstrations": len(demonstrations),
        "balanced_samples": len(balanced_samples),
        "timestamp": timestamp,
        "metric_type": "ai_feedback_enhanced"
    }
    
    info_path = os.path.join(output_dir, f"{model_name}_info.json")
    try:
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        print(f"📊 Training info saved to: {info_path}")
    except Exception as e:
        print(f"⚠️ Failed to save training info: {e}")
    
    print(f"\n✅ DSPy Agent Fine-tuning Pipeline Completed!")
    print(f"📁 Model saved in: {output_dir}")
    
    return model_path

def main():
    parser = argparse.ArgumentParser(description="DSPy Agent Fine-tuning Script")
    
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to dataset directory (e.g., 1_GT_collection/GT_dataset/dataset_x)")
    parser.add_argument("--teacher-model", type=str, default="llama-3.3-70b-versatile",
                       help="Teacher model ID (default: llama-3.3-70b-versatile)")
    parser.add_argument("--student-model", type=str, default="llama3.1:70b",
                       help="Student model ID to fine-tune (default: llama3.1:70b)")
    parser.add_argument("--teacher-groq", action="store_true", default=True,
                       help="Use GROQ API for teacher model (default: True)")
    parser.add_argument("--student-groq", action="store_true", default=False,
                       help="Use GROQ API for student model (default: False)")
    parser.add_argument("--assessor-model", type=str, default="deepseek-r1-distill-llama-70b",
                       help="Assessor model ID for AI feedback metric (default: deepseek-r1-distill-llama-70b)")
    parser.add_argument("--assessor-groq", action="store_true", default=True,
                       help="Use GROQ API for assessor model (default: True)")
    parser.add_argument("--hint-type", type=str, default="action",
                       choices=["action", "subgoal"],
                       help="Type of hint to predict (default: action)")
    parser.add_argument("--encoding-type", type=str, default="ascii",
                       choices=["ascii", "natural", "tuples", "relative"],
                       help="Text encoding type (default: ascii)")
    parser.add_argument("--samples-per-category", type=int, default=10,
                       help="Number of samples per category (default: 10)")
    parser.add_argument("--max-bootstrapped-demos", type=int, default=1,
                       help="Maximum number of bootstrapped demonstrations (default: 1)")
    parser.add_argument("--minibatch-size", type=int, default=40,
                       help="Minibatch size for optimization (default: 40)")
    parser.add_argument("--output-dir", type=str, default="saved_llm_models",
                       help="Directory to save trained models (default: saved_llm_models)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run fine-tuning pipeline
    model_path = run_dspy_agent_fine_tuning(
        dataset_path=args.dataset_path,
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        teacher_groq=args.teacher_groq,
        student_groq=args.student_groq,
        hint_type=args.hint_type,
        encoding_type=args.encoding_type,
        samples_per_category=args.samples_per_category,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        minibatch_size=args.minibatch_size,
        assessor_model=args.assessor_model,
        assessor_groq=args.assessor_groq,
        output_dir=args.output_dir
    )
    
    if model_path:
        print(f"\n🎉 Success! Fine-tuned model available at: {model_path}")
    else:
        print(f"\n⚠️ Process completed but model save failed. Check logs for details.")

if __name__ == "__main__":
    main() 