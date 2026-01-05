# DSPy Agent Fine-tuning

This module implements the DSPy teacher-student fine-tuning approach with AI feedback metrics for training intelligent agents on Minigrid navigation tasks.

## ✨ Recent Updates

- **Enhanced AI Feedback Metrics**: Uses third LLM for nuanced action evaluation
- **Improved Error Handling**: Robust error recovery and validation
- **Batch Processing**: Support for configuration-driven multiple model training
- **Clean Architecture**: Streamlined codebase with better documentation

## Overview

The fine-tuning process follows the approach described in the [DSPy Games Tutorial](https://dspy.ai/tutorials/games/) with enhanced AI feedback evaluation:

1. **Prompt Optimization**: Use a larger teacher model (default: `llama-3.3-70b-versatile` via GROQ) to optimize prompts
2. **Knowledge Distillation**: Use the optimized teacher to fine-tune a smaller student model (default: `llama3.1:70b` locally)
3. **AI Feedback Metric**: Use a third LLM (default: `deepseek-r1-distill-llama-70b`) to evaluate if incorrect predictions still make logical sense
4. **Model Saving**: Save the fine-tuned model locally for later use

### Enhanced Metric System

The evaluation metric uses AI feedback following [DSPy's metric patterns](https://dspy.ai/learn/evaluation/metrics/):

- **Exact Match**: If predicted action = ground truth → Score = 1.0
- **AI Assessment**: If different → Third LLM evaluates if prediction makes sense → Score = 0.5 if reasonable, 0.0 if not
- **Contextual Evaluation**: Considers current state, agent reasoning, and action semantics

## Usage

### Basic Usage

```bash
python dspy_agent_fine_tuning.py --dataset-path ../1_GT_collection/GT_dataset/dataset_9env_5seed_5episodes_v2_0623_2116
```

### Advanced Usage

```bash
python dspy_agent_fine_tuning.py \
  --dataset-path ../1_GT_collection/GT_dataset/dataset_9env_5seed_5episodes_v2_0623_2116 \
  --teacher-model llama-3.3-70b-versatile \
  --student-model llama3.1:70b \
  --teacher-groq \
  --assessor-model deepseek-r1-distill-llama-70b \
  --assessor-groq \
  --hint-type action \
  --encoding-type ascii \
  --samples-per-category 15 \
  --output-dir ./fine_tuned_models
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset-path` | *Required* | Path to the dataset directory |
| `--teacher-model` | `llama-3.3-70b-versatile` | Teacher model ID for prompt optimization |
| `--student-model` | `llama3.1:70b` | Student model ID to fine-tune |
| `--teacher-groq` | `True` | Use GROQ API for teacher model |
| `--student-groq` | `False` | Use GROQ API for student model (False = local Ollama) |
| `--assessor-model` | `deepseek-r1-distill-llama-70b` | Assessor model ID for AI feedback metric |
| `--assessor-groq` | `True` | Use GROQ API for assessor model |
| `--hint-type` | `action` | Type of hint to predict (`action` or `subgoal`) |
| `--encoding-type` | `ascii` | Text encoding type (`ascii`, `natural`, `tuples`, `relative`) |
| `--samples-per-category` | `10` | Number of samples per category |
| `--max-bootstrapped-demos` | `1` | Maximum bootstrapped demonstrations for MIPROv2 |
| `--minibatch-size` | `40` | Minibatch size for optimization |
| `--output-dir` | `saved_llm_models` | Directory to save trained models |

## Requirements

### API Keys
- **GROQ API Key**: Required for teacher model (set in `utils/load_key.py`)

### Local Models (Student)
- **Ollama**: For running local models like `llama3.1:70b`
  ```bash
  # Install Ollama and pull the model
  curl -fsSL https://ollama.ai/install.sh | sh
  ollama pull llama3.1:70b
  ```

## Output

The script generates:
- **Fine-tuned Model**: `{model_name}.pkl` - The trained DSPy agent
- **Training Info**: `{model_name}_info.json` - Metadata and training statistics
- **Console Logs**: Detailed progress and evaluation metrics

## Example Output

```
🎯 Starting DSPy Agent Fine-tuning Pipeline
============================================================
📁 Dataset: ../1_GT_collection/GT_dataset/dataset_9env_5seed_5episodes_v2_0623_2116
🧠 Teacher Model: llama-3.3-70b-versatile (GROQ: True)
🎓 Student Model: llama3.1:70b (GROQ: False)
🎯 Hint Type: action
📝 Encoding: ascii
============================================================

📊 Loading and preparing dataset...
Loaded 45 demonstrations
Available categories: {'0': 28, '1': 32, '2': 89, '3': 12, '4': 5, '5': 8, '6': 15}
Category '0': sampled 10/28 samples
Category '1': sampled 10/32 samples
...

🧠 Step 1: Optimizing prompts with teacher model (llama-3.3-70b-versatile)...
🔧 Running prompt optimization with MIPROv2...
✅ Prompt optimization completed successfully

🎓 Step 2: Setting up student model (llama3.1:70b)...

🔥 Step 3: Fine-tuning student model...
✅ Fine-tuning completed successfully

📊 Step 4: Evaluating fine-tuned model...
🤖 AI Feedback Metric initialized with deepseek-r1-distill-llama-70b
📈 Evaluation Accuracy: 0.850 (17/20)
  - Exact matches: 14/20 (0.70)
  - AI-assessed reasonable: 3/6 (0.50)

💾 Step 5: Saving fine-tuned model...
✅ Fine-tuned model saved to: saved_llm_models/finetuned_llama3_1_70b_action_ascii_20250125_143022.pkl
📊 Training info saved to: saved_llm_models/finetuned_llama3_1_70b_action_ascii_20250125_143022_info.json

✅ DSPy Agent Fine-tuning Pipeline Completed!
```

## Loading and Using Fine-tuned Models

```python
import dspy
from utils.dspy_signature import configure_llm

# Configure with automatic model selection based on source
configure_llm(source="local")  # Uses llama3.1:latest for local
# configure_llm(source="GROQ")   # Uses llama-3.3-70b-versatile for GROQ  
# configure_llm(source="OpenAI") # Uses openai/gpt-4o-mini for OpenAI

# Load the fine-tuned agent
agent = MinigridAgent()
agent.load("saved_llm_models/finetuned_llama3_1_70b_action_ascii_20250125_143022.pkl")

# Use the agent
observation = "The agent is in a room with a red door to the north..."
prediction = agent(observation=observation)
print(f"Predicted action: {prediction.primitive_action}")
```

## AI Feedback Metric Details

The enhanced metric system provides more nuanced evaluation than simple exact match:

### How It Works

1. **Exact Match (Score = 1.0)**: When predicted action equals ground truth
2. **AI Assessment (Score = 0.5 or 0.0)**: When actions differ, a third LLM evaluates:
   - **Input**: Current environment state, predicted action, agent's reasoning, expected action
   - **Question**: "Does the predicted action make logical sense as a reasonable alternative?"
   - **Output**: Boolean assessment of reasonableness

### Benefits

- **Captures Valid Alternatives**: Some navigation scenarios have multiple valid actions
- **Reduces Over-Penalization**: Reasonable but suboptimal actions get partial credit
- **Contextual Understanding**: Considers environment state and agent reasoning
- **Optimization Guidance**: Helps DSPy optimizers understand acceptable variations

### Example Scenarios

```
Scenario: Agent at (1,1), red door at (3,1), key at (2,1)
Ground Truth: "move forward" (to get key first)
Prediction: "turn right" (to face door directly)
AI Assessment: ✅ "Makes sense - alternative valid approach"
Score: 0.5 instead of 0.0
```

### Customization

```bash
# Use different assessor model
--assessor-model llama-3.3-70b-versatile

# Use local assessor model
--assessor-model llama3.1:70b --no-assessor-groq
```

## Configuration File Usage

The pipeline supports YAML configuration files for batch processing multiple models:

### Basic Config Structure

```yaml
fine_tuning:
  dataset_path: "path/to/dataset"
  approach: "teacher_student"
  output_dir: "saved_llm_models"
  
  defaults:
    teacher_model: "llama-3.3-70b-versatile"
    student_model: "llama3.1:70b"
    assessor_model: "deepseek-r1-distill-llama-70b"
    teacher_groq: true
    student_groq: false
    assessor_groq: true
  
  models:
    - name: "action_ascii_experiment"
      hint_type: "action"
      encoding_type: "ascii"
      samples_per_category: 15
```

### Running with Config

```bash
# Use config file (dataset path can be overridden)
python main_llm_optimization.py \
  --config configs/llm_optimization_config.yaml \
  --dataset-path ../1_GT_collection/GT_dataset/dataset_x
```

## Troubleshooting

### Common Issues

1. **GROQ Rate Limits**: The script includes automatic retry with exponential backoff
2. **Ollama Connection**: Ensure Ollama is running (`ollama serve`)
3. **Memory Issues**: Reduce `--samples-per-category` or `--minibatch-size`
4. **Model Save Errors**: The script falls back to saving metadata if model save fails

### Performance Tips

- **Start Small**: Use `--samples-per-category 5` for initial testing
- **Encoding Choice**: Use `--encoding-type ascii` for faster processing
- **API Management**: Monitor GROQ API usage to avoid rate limits
- **Batch Processing**: Use config files for multiple model training
- **Error Recovery**: The pipeline includes automatic fallbacks and retry mechanisms

## Integration with Other Modules

This fine-tuned model can be used with:
- **3_LLM_evaluation**: For evaluation using `--model-path`
- **5_RL_agent_comparison**: As a hint source in `HintWrapper`
- **Interactive testing**: Using `interactive_llm_tester.py` 