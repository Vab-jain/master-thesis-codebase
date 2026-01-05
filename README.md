# Master Thesis Codebase

This repository contains the complete codebase for the master's thesis research on integrating Large Language Models (LLMs) with Reinforcement Learning (RL) agents across multiple game environments. The research explores how LLM-generated hints can improve RL agent training efficiency and performance.

## 📝 Abstract

This thesis investigates how Large Language Models (LLMs) can guide exploration in Reinforcement Learning (RL) without imposing hard constraints on agent decision-making. We hypothesize that LLMs encode broad world knowledge and commonsense procedural regularities that, when elicited via prompting, can bias exploration while preserving policy autonomy. To investigate this, we propose a soft-constraint integration wherein LLM-generated suggestions are provided as structured hints within the agent’s observation, accompanied by a hint-availability flag. This design preserves the original Markov Decision Process (MDP), leaves the learning objective unchanged, and allows the agent's policy to learn when to use or ignore guidance.

Methodologically, we develop a prompting and encoding pipeline to translate compact state summaries into schema-constrained outputs that can be consumed by standard RL policies. The approach is algorithm-agnostic; we instantiate it with Proximal Policy Optimization (PPO) and evaluate across domains of varying structure and difficulty: Minigrid, TicTacToe, and Deal or No Deal. Supplementary experiments with DQN and REINFORCE are also provided.

Empirical results demonstrate that hints elicited via structure-preserving prompts—augmented with chain-of-thought reasoning where appropriate—are reliable and context-relevant. In Minigrid, integrating LLM hints as soft inputs yields improved sample efficiency and, on more difficult tasks, better final performance relative to tabula-rasa baselines, while remaining below an oracle upper bound. In compact domains such as TicTacToe and Deal or No Deal, the prompting pipeline produces interpretable, valid suggestions (e.g., higher action validity under masking and approximately 85% agreement with curated data in Deal or No Deal), though overall training gains are bounded by the short horizon and small state spaces. In all settings, RL agents learn to discount suboptimal hints, showcasing robustness to imperfect guidance.

We discuss key limitations, particularly the computational overhead of frequent LLM queries, and outline cost-aware extensions—including adaptive hint scheduling, distillation, and lightweight serving. Overall, our results support LLM-guided hints as a practical and robust mechanism for accelerating learning in sufficiently complex RL tasks while preserving agent autonomy.


## 📁 Repository Structure

```
master-thesis-codebase/
├── deal_or_no_deal/          # Deal or No Deal negotiation game RL-LLM integration
├── minigrid-rl-llm/          # BabyAI/Minigrid environments RL-LLM research pipeline
├── tictactoe_rl_llm/         # Tic-Tac-Toe game RL-LLM integration
└── thesis.pdf                # Complete thesis document
```

## 🎯 Overview

This codebase contains the implementation of a unified research thesis that investigates the integration of LLMs with RL agents, applying the same methodology across three different environments:

1. **Deal or No Deal**: Multi-issue bargaining negotiation environment with RL agents trained using PPO and REINFORCE algorithms, enhanced with LLM-generated hints.

2. **Minigrid-RL-LLM**: Comprehensive research pipeline for BabyAI/Minigrid environments, including ground truth data collection, LLM optimization, evaluation, RL training, and agent comparison.

3. **Tic-Tac-Toe RL-LLM**: Simple grid-based game environment exploring LLM-guided RL training with various prompting strategies and board representations.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Conda (recommended for environment management)
- CUDA-capable GPU (recommended for training)

### Installation

Each project has its own requirements. Navigate to the specific project directory and install dependencies:

```bash
# For Deal or No Deal
cd deal_or_no_deal
pip install -r requirements.txt

# For Minigrid-RL-LLM
cd minigrid-rl-llm
pip install -r requirements.txt

# For Tic-Tac-Toe RL-LLM
cd tictactoe_rl_llm
pip install -r requirements.txt
```

## 📚 Project Details

### 1. Deal or No Deal (`deal_or_no_deal/`)

A Gym-compatible environment for the Deal or No Deal negotiation task, implementing multi-issue bargaining with coarse dialogue acts. The project includes:

- **Environment**: Custom Gymnasium environment for negotiation scenarios
- **Algorithms**: PPO and REINFORCE implementations
- **LLM Integration**: Hint injection system using LLM-generated guidance
- **Evaluation**: Comprehensive metrics and policy evaluation tools

**Key Features:**
- Multi-issue negotiation over 3 item types (books, hats, balls)
- Coarse dialogue acts: propose, insist, agree, disagree, end
- Support for LLM-generated hints during training
- Supervised expert training baseline

**Quick Start:**
```bash
cd deal_or_no_deal
conda activate rl  # As per project setup
python train/train.py --config configs/ppo_config.yaml
```

**Documentation:**
- Environment details: `deal_or_no_deal/deal_or_no_deal_env/README.md`
- Dialog system: `deal_or_no_deal/deal_or_no_dialog/README.md`

### 2. Minigrid-RL-LLM (`minigrid-rl-llm/`)

A comprehensive research pipeline for studying LLM-RL integration in BabyAI/Minigrid environments. The project is organized into sequential modules:

**Module Structure:**
- `1_GT_collection/`: Ground truth data collection from BabyAI bot
- `2_LLM_optimization/`: LLM fine-tuning and ablation studies using DSPy
- `3_LLM_evaluation/`: Evaluation of trained LLM models
- `4_RL_agent_training/`: RL agent training with Stable Baselines3
- `5_RL_agent_comparison/`: Agent comparison and analysis
- `6_CoP_Approach/`: Chain-of-Thought approach implementation

**Key Features:**
- Multiple observation encodings (natural language, ASCII, tuples, relative)
- DSPy-based LLM optimization for subgoal and action prediction
- Hint system with configurable frequency and sources
- Comprehensive evaluation and comparison tools

**Quick Start:**
```bash
cd minigrid-rl-llm
# Follow the pipeline sequentially
cd 1_GT_collection && python gt_data_collection.py
cd ../2_LLM_optimization && python main_llm_optimization.py
# ... and so on
```

**Documentation:**
- Main README: `minigrid-rl-llm/README.md`
- Module-specific READMEs in each numbered directory

### 3. Tic-Tac-Toe RL-LLM (`tictactoe_rl_llm/`)

A simplified grid-based game environment exploring LLM-guided RL training with various configurations:

**Key Features:**
- Multiple board representations (1D, 2D, etc.)
- Various prompting methods (Zero-Shot, Chain-of-Thought, etc.)
- LLM suggestion integration with configurable probability
- Ground truth database for comparison
- Comprehensive experiment tracking

**Quick Start:**
```bash
cd tictactoe_rl_llm
# Run RL experiments
python run_RL_experiments.py

# Run LLM experiments
python run_LLM_experiment.py
```

## 🔬 Research Methodology

All three projects follow a similar research methodology:

1. **Baseline Establishment**: Train RL agents without LLM assistance
2. **LLM Integration**: Incorporate LLM-generated hints/guidance during training
3. **Evaluation**: Compare sample efficiency and final performance
4. **Analysis**: Statistical analysis and visualization of results

## 📊 Results and Outputs

Each project generates:
- Trained model checkpoints
- Training logs and metrics (CSV, JSON)
- Evaluation results
- Comparison plots and visualizations
- Configuration snapshots

Results are stored in project-specific directories:
- `deal_or_no_deal/runs/`
- `minigrid-rl-llm/4_RL_agent_training/RL_Training_Results_*/`
- `tictactoe_rl_llm/experiments_results/`

## 🛠️ Common Dependencies

While each project has specific requirements, common dependencies include:
- PyTorch
- Gymnasium
- NumPy, Pandas, Matplotlib
- PyYAML
- tqdm

## 🔗 References

- **Thesis**: [Full thesis PDF](./thesis.pdf)

- **Integrating Large Language Models with RL Agents: Best Practices and Benchmarks.** [arXiv:2510.08779](https://arxiv.org/abs/2510.08779)

## 👤 Author

**Vaibhav Jain**

Master's Thesis Research (In fullfilment of the MS Data Science and AI Program)

University of Saarland (UdS)

---

For detailed documentation on each project, please refer to the README files in the respective project directories.

Contact: vaja00001@uni-saarland.de (or create an issue on this repo)

