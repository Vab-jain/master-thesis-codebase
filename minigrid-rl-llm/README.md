# BabyAI RL-LLM Research Codebase

This codebase implements a comprehensive research pipeline for studying the integration of Large Language Models (LLMs) with Reinforcement Learning (RL) agents in BabyAI environments. The system supports ground truth data collection, LLM optimization, evaluation, RL agent training, and comparison analysis.

## 🏗️ Codebase Structure

The codebase is organized into five main modules:

```
minigrid-rl-llm/
├── 1_GT_collection/              # Ground Truth Data Collection
│   ├── GT_dataset/               # Collected demonstration data
│   └── gt_data_collection.py     # Main GT collection script
├── 2_LLM_optimization/           # LLM Optimization & Fine-tuning
│   ├── saved_llm_models/         # Trained DSPy models
│   ├── ablation_study.py         # Ablation study implementation
│   ├── fine_tune_models.py       # Model fine-tuning
│   ├── analyze_ablation_results.py # Results analysis
│   └── main_llm_optimization.py  # Main optimization script
├── 3_LLM_evaluation/             # LLM Model Evaluation
│   ├── LLM_evaluation/           # Evaluation results
│   ├── llm_evaluation.py         # Evaluation implementation
│   └── main_llm_evaluation.py    # Main evaluation script
├── 4_RL_agent_training/          # RL Agent Training
│   ├── RL_Trained_Agents/        # Trained RL agents
│   ├── train_stable_baselines.py # Stable Baselines training
│   ├── train_with_hints.py       # Hint-based training
│   └── main_rl_training.py       # Main training script
├── 5_RL_agent_comparison/        # RL Agent Comparison
│   ├── RL_Agents_Comparison/     # Comparison results
│   └── main_agent_comparison.py  # Main comparison script
├── utils/                        # Shared utilities
│   ├── observation_encoder.py    # Text encoding utilities
│   └── hint_wrapper.py           # Hint wrapper implementation
├── configs/                      # Configuration files
│   ├── gt_collection_config.yaml
│   ├── llm_optimization_config.yaml
│   ├── llm_evaluation_config.yaml
│   ├── rl_training_config.yaml
│   └── agent_comparison_config.yaml
└── sand_*.py                     # Sandboxed test files
```

## 🚀 Quick Start

### 1. Ground Truth Data Collection

Collect BabyAI bot demonstrations with text encodings:

```bash
cd 1_GT_collection
python gt_data_collection.py --env-ids BabyAI-GoToObj-v0 BabyAI-GoToLocal-v0 --seeds 42 123
```

### 2. LLM Optimization

Run ablation studies and fine-tuning:

```bash
cd 2_LLM_optimization
python main_llm_optimization.py --mode all
```

### 3. LLM Evaluation

Evaluate trained DSPy models:

```bash
cd 3_LLM_evaluation
python main_llm_evaluation.py --model-path ../2_LLM_optimization/saved_llm_models/subgoal_predictor_ascii.json
```

### 4. RL Agent Training

Train RL agents with or without hints:

```bash
cd 4_RL_agent_training
python main_rl_training.py --env-id BabyAI-GoToObj-v0 --use-hints --hint-type subgoal
```

### 5. Agent Comparison

Compare trained agents:

```bash
cd 5_RL_agent_comparison
python main_agent_comparison.py --pattern "*_no_hints_*" "*_subgoal_freq5_*"
```

## 📋 Detailed Usage

### Ground Truth Collection

The GT collection module collects demonstrations from the BabyAI bot and adds multiple text encodings:

- **Natural Language**: Human-readable descriptions
- **ASCII Grid**: Visual grid representation
- **Tuple Lists**: Structured object lists
- **Relative Descriptions**: Position-relative descriptions

```bash
# Using configuration file
python gt_data_collection.py --config ../configs/gt_collection_config.yaml

# Using command line arguments
python gt_data_collection.py \
  --env-ids BabyAI-GoToObj-v0 BabyAI-GoToLocal-v0 \
  --seeds 42 123 456 \
  --output-dir GT_dataset
```

### LLM Optimization

The LLM optimization module supports:

- **Ablation Studies**: Compare different encoding types and bootstrapped examples
- **Fine-tuning**: Train DSPy models for subgoal and action prediction
- **Analysis**: Generate comparison plots and statistics

```bash
# Run complete pipeline
python main_llm_optimization.py --mode all

# Run specific components
python main_llm_optimization.py --mode ablation
python main_llm_optimization.py --mode fine_tuning
python main_llm_optimization.py --mode analysis
```

### LLM Evaluation

Evaluate trained DSPy models on test datasets:

```bash
# Evaluate single model
python main_llm_evaluation.py \
  --model-path ../2_LLM_optimization/saved_llm_models/subgoal_predictor_ascii.json \
  --env-id BabyAI-GoToObj-v0 \
  --encoding-type ascii \
  --hint-type subgoal

# Run multiple evaluations from config
python main_llm_evaluation.py --config ../configs/llm_evaluation_config.yaml
```

### RL Agent Training

Train RL agents using Stable Baselines3 with optional hints:

```bash
# Train baseline agent (no hints)
python main_rl_training.py \
  --env-id BabyAI-GoToObj-v0 \
  --obs-type multi \
  --total-timesteps 50000

# Train agent with BabyAI bot hints
python main_rl_training.py \
  --env-id BabyAI-GoToObj-v0 \
  --obs-type multi \
  --use-hints \
  --hint-type subgoal \
  --hint-frequency 5 \
  --hint-source babyai_bot \
  --total-timesteps 50000

# Train agent with DSPy hints
python main_rl_training.py \
  --env-id BabyAI-GoToObj-v0 \
  --obs-type multi \
  --use-hints \
  --hint-type subgoal \
  --hint-frequency 5 \
  --hint-source dspy \
  --hint-model-path ../2_LLM_optimization/saved_llm_models/subgoal_predictor_ascii.json \
  --total-timesteps 50000

# Run multiple experiments from config
python main_rl_training.py --config ../configs/rl_training_config.yaml
```

### Agent Comparison

Compare trained agents and generate analysis plots:

```bash
# Compare specific agents
python main_agent_comparison.py \
  --experiment-dirs \
    ../4_RL_agent_training/RL_Trained_Agents/BabyAI_GoToObj_v0_multi_no_hints_50000ts_seed42 \
    ../4_RL_agent_training/RL_Trained_Agents/BabyAI_GoToObj_v0_multi_subgoal_freq5_babyai_bot_50000ts_seed42

# Compare using patterns
python main_agent_comparison.py \
  --pattern "*_no_hints_*" "*_subgoal_freq5_*" \
  --plot-title "Baseline vs Subgoal Hints"

# Run predefined comparisons from config
python main_agent_comparison.py --config ../configs/agent_comparison_config.yaml
```

## 🔧 Configuration Files

Each module has a corresponding YAML configuration file in the `configs/` directory:

- `gt_collection_config.yaml`: GT data collection settings
- `llm_optimization_config.yaml`: LLM optimization parameters
- `llm_evaluation_config.yaml`: LLM evaluation settings
- `rl_training_config.yaml`: RL training experiments
- `agent_comparison_config.yaml`: Agent comparison scenarios

## 🎯 Hint System

The hint system provides two types of hints to RL agents:

### Hint Types
- **Subgoal Hints**: High-level planning hints (e.g., "go to the red key")
- **Action Hints**: Low-level control hints (e.g., "turn left", "move forward")

### Hint Sources
- **BabyAI Bot**: Oracle hints from the BabyAI bot
- **DSPy Models**: Learned hints from fine-tuned DSPy models

### Hint Configuration
- **Frequency**: How often hints are provided (every k steps)
- **Probability**: Probability of providing a hint when due
- **Encoding**: Text encoding type for DSPy hints

## 📊 Output and Results

### GT Collection
- JSON files with demonstrations and text encodings
- Organized by environment and seed

### LLM Optimization
- Trained DSPy models in JSON format
- Ablation study results with plots
- Performance comparison tables

### LLM Evaluation
- Accuracy metrics for each model
- Detailed evaluation reports
- Cross-environment performance analysis

### RL Training
- Trained PPO models with training logs
- Evaluation results and best models
- Training configuration and metadata

### Agent Comparison
- Sample efficiency plots (raw and smoothed)
- Training time comparisons
- Final performance analysis
- Statistical significance tests

## 🛠️ Dependencies

```bash
pip install gymnasium minigrid dspy-ai stable-baselines3 torch matplotlib pandas numpy pyyaml
```

## 📝 Notes

- All scripts support both configuration files and command-line arguments
- Results are automatically saved with timestamps
- The hint wrapper supports both vectorized and non-vectorized environments
- Sample efficiency plots include both raw and smoothed curves
- All experiments include proper logging and error handling

## 🔄 Pipeline Workflow

1. **Collect GT Data**: Gather demonstrations from BabyAI bot
2. **Optimize LLMs**: Run ablation studies and fine-tune DSPy models
3. **Evaluate LLMs**: Test model performance on held-out data
4. **Train RL Agents**: Train agents with and without hints
5. **Compare Agents**: Analyze sample efficiency and final performance

This modular design allows for independent experimentation and easy extension of the research pipeline.