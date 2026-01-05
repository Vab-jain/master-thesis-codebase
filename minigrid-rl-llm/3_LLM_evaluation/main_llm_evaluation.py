#!/usr/bin/env python3
"""
Main LLM Evaluation Script
Evaluates DSPy models on unified test datasets.
Results are saved in LLM_evaluation/ directory.
"""

import os
import sys
import argparse
import yaml
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import Counter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_evaluation import Evaluate

class LLMEvaluator:
    """Main class for evaluating LLM models."""
    
    def __init__(self, output_dir: str = "LLM_evaluation"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single model with given configuration."""
        print(f"\n🔍 Evaluating LLM Model")
        print(f"Dataset: {config.get('dataset_path', 'N/A')}")
        print(f"Hint type: {config.get('hint_type', 'N/A')}")
        print(f"Encoding type: {config.get('encoding_type', 'N/A')}")
        print(f"Model path: {config.get('model_path', 'Direct LLM')}")
        print(f"Test set size: {config.get('testset_size', 'N/A')}")
        
        start_time = time.time()
        
        # Create evaluator
        evaluator = Evaluate(
            dataset_path=config.get('dataset_path'),
            hint_type=config.get('hint_type', 'subgoal'),
            encoding_type=config.get('encoding_type', 'ascii'),
            model_path=config.get('model_path'),  # Can be None for direct LLM
            testset_size=config.get('testset_size', 10)
        )
        
        # Run evaluation
        results = evaluator.evaluate()
        
        # Add metadata
        results.update({
            "evaluation_time_seconds": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "config": config
        })
        
        return results
    
    def run_evaluation_experiments(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run multiple evaluation experiments."""
        print(f"🎯 Starting {len(configs)} evaluation experiments...")
        
        all_results = []
        
        for i, config in enumerate(configs, 1):
            print(f"\n{'='*60}")
            print(f"EVALUATION {i}/{len(configs)}")
            print(f"{'='*60}")
            
            try:
                results = self.evaluate_model(config)
                all_results.append(results)
                
                # Save individual results
                experiment_name = self._generate_experiment_name(config)
                experiment_output_dir = config.get('output_dir', self.output_dir)
                experiment_dir = os.path.join(experiment_output_dir, experiment_name)
                os.makedirs(experiment_dir, exist_ok=True)
                
                # Save results
                results_file = os.path.join(experiment_dir, "evaluation_results.json")
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"✅ Evaluation completed!")
                print(f"📁 Results saved to: {experiment_dir}")
                
            except Exception as e:
                print(f"❌ Evaluation {i} failed: {e}")
                continue
        
        return all_results
    
    def _generate_experiment_name(self, config: Dict[str, Any]) -> str:
        """Generate descriptive experiment name."""
        hint_type = config.get('hint_type', 'unknown')
        encoding_type = config.get('encoding_type', 'unknown')
        testset_size = config.get('testset_size', 'unknown')
        
        if config.get('model_path'):
            model_name = os.path.basename(config['model_path']).replace('.json', '')
            return f"eval_{hint_type}_{encoding_type}_{model_name}_{testset_size}samples"
        else:
            return f"eval_{hint_type}_{encoding_type}_direct_llm_{testset_size}samples"
    
    def create_evaluation_summary(self, results: List[Dict[str, Any]]) -> None:
        """Create a summary of all evaluation results."""
        if not results:
            print("No results to summarize!")
            return
        
        print("\n" + "="*80)
        print("LLM EVALUATION SUMMARY")
        print("="*80)
        
        summary_data = []
        
        for i, result in enumerate(results, 1):
            config = result.get("config", {})
            
            # Create descriptive name
            experiment_name = self._generate_experiment_name(config)
            
            # Extract metrics
            accuracy = result.get('accuracy', 0) * 100
            evaluation_time = result.get('evaluation_time_seconds', 0)
            
            summary_data.append({
                "experiment_name": experiment_name,
                "accuracy": accuracy,
                "evaluation_time": evaluation_time,
                "config": config
            })
            
            print(f"\n📊 EVALUATION {i}: {experiment_name}")
            print(f"  Dataset:         {config.get('dataset_path', 'N/A')}")
            print(f"  Hint type:       {config.get('hint_type', 'N/A')}")
            print(f"  Encoding:        {config.get('encoding_type', 'N/A')}")
            print(f"  Model:           {config.get('model_path', 'Direct LLM')}")
            print(f"  Test set size:   {config.get('testset_size', 'N/A')}")
            print(f"  Accuracy:        {accuracy:.1f}%")
            print(f"  Evaluation time: {evaluation_time:.1f} seconds")
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(self.output_dir, f"evaluation_summary_{timestamp}.json")
        
        with open(summary_file, 'w') as f:
            json.dump({
                "summary": summary_data,
                "timestamp": datetime.now().isoformat(),
                "total_evaluations": len(results)
            }, f, indent=2)
        
        print(f"\n📁 Summary saved to: {summary_file}")
        print("\n" + "="*80)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Main LLM Evaluation Script")
    
    parser.add_argument("--config", type=str, default="configs/llm_evaluation_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--dataset-path", type=str, required=False,
                       help="Path to dataset directory (e.g., GT_dataset/dataset_x)")
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
    parser.add_argument("--output-dir", type=str, default="LLM_evaluation",
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = LLMEvaluator(args.output_dir)
    
    # Load configuration or use command line arguments
    if os.path.exists(args.config):
        config_data = load_config(args.config)
        configs = config_data.get("evaluations", [])
        # Override dataset path from command line if provided
        if args.dataset_path:
            for config in configs:
                config["dataset_path"] = args.dataset_path
    else:
        # Create single evaluation config from command line arguments
        if not args.dataset_path:
            raise ValueError("Dataset path must be provided either via config file or command line")
        config = {
            "dataset_path": args.dataset_path,
            "hint_type": args.hint_type,
            "encoding_type": args.encoding_type,
            "model_path": args.model_path,
            "testset_size": args.testset_size
        }
        configs = [config]
    
    # Run evaluation experiments
    results = evaluator.run_evaluation_experiments(configs)
    
    # Create summary
    evaluator.create_evaluation_summary(results)
    
    print(f"\n🎉 Evaluation completed!")
    print(f"📁 Results saved in: {args.output_dir}")
    print(f"✅ Successfully evaluated {len(results)} models")

if __name__ == "__main__":
    main() 