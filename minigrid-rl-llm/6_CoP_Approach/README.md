# Chain-of-Planning (CoP) Approach for MiniGrid Navigation

This directory contains the evaluation and training data generation for LLM-based planners in MiniGrid environments.

## 📁 Directory Contents

### 🎯 Core Training Materials

- **`comprehensive_llm_training.txt`** (23KB)
  - Complete LLM training dataset with 28 expert examples
  - Combines simple navigation and complex task scenarios
  - Ready-to-use prompt for training LLM planners

- **`create_llm_examples.py`** (16KB)  
  - Production script for generating training examples
  - Uses BabyAI Bot (expert) and smart navigation policies
  - Handles both simple and complex environments

- **`simple_navigation_examples.txt`** (12KB)
  - Basic navigation examples for MiniGrid-Empty environments
  - Demonstrates obstacle avoidance and goal-seeking

- **`babyai_complex_examples.txt`** (9KB)
  - Complex task examples from BabyAI environments
  - Shows multi-step reasoning (keys → doors → goals)

### 📊 Evaluation Scripts & Results

- **`evaluate_planner.py`** (7KB)
  - Main evaluation script for testing planners
  - Tests on MiniGrid-Empty-5x5-v0 (partially observable)

- **`test_fully_observable.py`** (7KB)
  - Evaluation script for fully observable environments  
  - Used to isolate partial observability as a factor

- **`planner_adapter.py`** (3KB)
  - Adapter for converting fully observable observations
  - Formats data for planners expecting different input

- **`fully_observable_results_20250629_145954.json`** (702B)
  - Evaluation results from fully observable testing
  - Documents 0% success rate even with perfect information

## 🚀 Usage

### Generate Training Examples
```bash
python create_llm_examples.py
```

### Evaluate a Planner
```bash
python evaluate_planner.py
```

### Test with Full Observability
```bash
python test_fully_observable.py
```

## 📋 Key Findings

### Original LLM Planner Evaluation
- **Partially Observable**: 0% success (0/100 episodes), 100 avg steps
- **Fully Observable**: 0% success (0/50 episodes), 50 avg steps  
- **Conclusion**: LLM-generated planner failed due to fundamental algorithmic flaws, not partial observability

### Training Data Generated
- **28 total examples**: 16 simple navigation + 12 complex tasks
- **Expert-quality**: All actions from proven policies (BabyAI Bot + smart navigation)
- **Comprehensive coverage**: Various scenarios, orientations, and obstacle configurations

## 🎓 Research Implications

1. **LLM Limitations**: Current LLM code generation insufficient for spatial reasoning tasks
2. **Training Data Value**: High-quality expert examples crucial for LLM planner training  
3. **Evaluation Framework**: Robust testing needed across multiple observability conditions
4. **Alternative Approaches**: Oracle/expert demonstrations more reliable than LLM-generated policies

## 📄 File Dependencies

- Requires `minigrid` and `gymnasium` packages
- BabyAI Bot requires access to `../utils/baby_ai_bot.py`
- Evaluation scripts work with any planner implementing the standard interface

---

*Generated as part of thesis research on LLM-guided reinforcement learning in MiniGrid environments.* 