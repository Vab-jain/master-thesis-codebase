#!/usr/bin/env python3
"""
Main LLM Optimization Script

Fine-tunes DSPy agents using teacher-student approach with AI feedback metrics.
- Prompt optimization with larger teacher model (GROQ API)
- Knowledge distillation to smaller student model (local/GROQ)
- Enhanced evaluation with AI feedback assessment
- Batch processing support via configuration files

Results saved in saved_llm_models/ directory.
"""

import os
import sys
import argparse
import yaml
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dspy_agent_fine_tuning import run_dspy_agent_fine_tuning

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not isinstance(config, dict):
            raise ValueError(f"Invalid config format: expected dict, got {type(config)}")
        
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file {config_path}: {e}")

def run_fine_tuning_experiments(config: Dict[str, Any]) -> None:
    """Run fine-tuning experiments with different configurations."""
    print("🎯 Starting DSPy Teacher-Student Fine-tuning Pipeline")
    
    # Extract fine-tuning parameters
    fine_tune_config = config.get("fine_tuning", {})
    dataset_path = fine_tune_config.get("dataset_path")
    
    # Validate configuration
    if not dataset_path:
        raise ValueError("No dataset path specified in configuration!")
    
    # Verify dataset exists
    train_path = os.path.join(dataset_path, "train.json")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training dataset not found at: {train_path}")
    
    models_config = fine_tune_config.get("models", [])
    if not models_config:
        raise ValueError("No model configurations specified!")
    
    defaults = fine_tune_config.get("defaults", {})
    output_dir = fine_tune_config.get("output_dir", "saved_llm_models")
    
    print(f"📁 Dataset: {dataset_path}")
    print(f"💾 Output Directory: {output_dir}")
    print(f"🔧 Found {len(models_config)} model configurations")
    
    # Run fine-tuning for each configuration
    successful_runs = 0
    for i, model_config in enumerate(models_config, 1):
        model_name = model_config.get('name', f'model_{i}')
        print(f"\n{'='*60}")
        print(f"[{i}/{len(models_config)}] FINE-TUNING: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Merge defaults with model-specific config
            final_config = {**defaults, **model_config}
            
            result_path = run_dspy_agent_fine_tuning(
                dataset_path=dataset_path,
                teacher_model=final_config.get("teacher_model", "llama-3.3-70b-versatile"),
                student_model=final_config.get("student_model", "llama3.1:70b"),
                teacher_groq=final_config.get("teacher_groq", True),
                student_groq=final_config.get("student_groq", False),
                hint_type=final_config.get("hint_type", "action"),
                encoding_type=final_config.get("encoding_type", "ascii"),
                samples_per_category=final_config.get("samples_per_category", 10),
                max_bootstrapped_demos=final_config.get("max_bootstrapped_demos", 1),
                minibatch_size=final_config.get("minibatch_size", 40),
                assessor_model=final_config.get("assessor_model", "deepseek-r1-distill-llama-70b"),
                assessor_groq=final_config.get("assessor_groq", True),
                output_dir=output_dir
            )
            
            if result_path:
                print(f"✅ Successfully fine-tuned: {model_name}")
                successful_runs += 1
            else:
                print(f"⚠️ Fine-tuning completed with issues: {model_name}")
            
        except Exception as e:
            print(f"❌ Fine-tuning failed for {model_name}: {e}")
            continue
    
    print(f"\n🎉 Pipeline completed: {successful_runs}/{len(models_config)} models fine-tuned successfully")

def main():
    parser = argparse.ArgumentParser(description="Main LLM Optimization Script")
    
    parser.add_argument("--config", type=str, default="configs/llm_optimization_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to dataset directory (e.g., GT_dataset/dataset_x)")
    parser.add_argument("--approach", type=str, default="teacher_student",
                       choices=["teacher_student"],
                       help="Fine-tuning approach: teacher_student (DSPy approach)")
    parser.add_argument("--hint-type", type=str, default="action",
                       choices=["subgoal", "action"],
                       help="Type of hint to predict (action recommended for teacher_student approach)")
    parser.add_argument("--encoding-type", type=str, default="ascii",
                       choices=["ascii", "natural", "tuples", "relative"],
                       help="Text encoding type")
    parser.add_argument("--max-bootstrapped-demos", type=int, default=1,
                       help="Maximum number of bootstrapped demonstrations (1 recommended for teacher_student approach)")
    parser.add_argument("--samples-per-category", type=int, default=10,
                       help="Number of samples to query per category")
    parser.add_argument("--teacher-model", type=str, default="llama-3.3-70b-versatile",
                       help="Teacher model for teacher_student approach")
    parser.add_argument("--student-model", type=str, default="llama3.1:70b",
                       help="Student model for teacher_student approach")
    parser.add_argument("--assessor-model", type=str, default="deepseek-r1-distill-llama-70b",
                       help="Assessor model for AI feedback metric")
    parser.add_argument("--output-dir", type=str, default="saved_llm_models",
                       help="Directory to save trained models")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration or create from command line arguments
    if os.path.exists(args.config):
        try:
            config = load_config(args.config)
            # Override dataset path from command line
            if "fine_tuning" not in config:
                config["fine_tuning"] = {}
            config["fine_tuning"]["dataset_path"] = args.dataset_path
            print(f"✅ Using configuration file: {args.config}")
        except Exception as e:
            print(f"❌ Failed to load config file {args.config}: {e}")
            print("Falling back to command line arguments...")
            config = None
    else:
        print(f"⚠️ Config file {args.config} not found, using command line arguments")
        config = None
    
    # Create config from command line if needed
    if config is None:
        model_config = {
            "name": f"cli_{args.hint_type}_{args.encoding_type}_{args.max_bootstrapped_demos}demos",
            "hint_type": args.hint_type,
            "encoding_type": args.encoding_type,
            "max_bootstrapped_demos": args.max_bootstrapped_demos,
            "samples_per_category": args.samples_per_category,
            "teacher_model": args.teacher_model,
            "student_model": args.student_model,
            "teacher_groq": True,
            "student_groq": False,
            "assessor_model": args.assessor_model,
            "assessor_groq": True,
            "minibatch_size": 40
        }
        
        config = {
            "fine_tuning": {
                "dataset_path": args.dataset_path,
                "approach": args.approach,
                "output_dir": args.output_dir,
                "models": [model_config]
            }
        }
    
    print("🚀 LLM Optimization Pipeline")
    print("=" * 50)
    print(f"📁 Dataset: {args.dataset_path}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🔧 Approach: {args.approach}")
    if args.approach == "teacher_student":
        print(f"🧠 Teacher Model: {args.teacher_model}")
        print(f"🎓 Student Model: {args.student_model}")
    
    # Run fine-tuning experiments
    run_fine_tuning_experiments(config)
    
    print(f"\n✅ LLM optimization completed!")
    print(f"📁 Models saved in: {args.output_dir}")

if __name__ == "__main__":
    main() 