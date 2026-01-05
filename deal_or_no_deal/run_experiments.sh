#!/bin/bash
# Experiment commands for PPO and REINFORCE with different seeds and hint modes

# PPO experiments
python -m train.train --config configs/ppo_config.yaml --algo ppo --hint none --seed 42 --output_dir "runs/ppo_small/ppo_seed42_3000steps"
python -m train.train --config configs/ppo_config.yaml --algo ppo --hint none --seed 46 --output_dir "runs/ppo_small/ppo_seed46_3000steps"
python -m train.train --config configs/ppo_config.yaml --algo ppo --hint none --seed 123 --output_dir "runs/ppo_small/ppo_seed123_3000steps"
python -m train.train --config configs/ppo_config.yaml --algo ppo --hint none --seed 412 --output_dir "runs/ppo_small/ppo_seed412_3000steps"
python -m train.train --config configs/ppo_config.yaml --algo ppo --hint none --seed 512 --output_dir "runs/ppo_small/ppo_seed512_3000steps"

# REINFORCE experiments
python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint none --seed 42 --output_dir "runs/reinforce_small/reinforce_seed42_2000steps"
python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint none --seed 46 --output_dir "runs/reinforce_small/reinforce_seed46_2000steps"
python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint none --seed 123 --output_dir "runs/reinforce_small/reinforce_seed123_2000steps"
python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint none --seed 412 --output_dir "runs/reinforce_small/reinforce_seed412_2000steps"
python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint none --seed 512 --output_dir "runs/reinforce_small/reinforce_seed512_2000steps"


# PPO experiments - random hints
python -m train.train --config configs/ppo_config.yaml --algo ppo --hint random --seed 42 --output_dir "runs/ppo_small/ppo_random_seed42_3000steps"
python -m train.train --config configs/ppo_config.yaml --algo ppo --hint random --seed 46 --output_dir "runs/ppo_small/ppo_random_seed46_3000steps"
python -m train.train --config configs/ppo_config.yaml --algo ppo --hint random --seed 123 --output_dir "runs/ppo_small/ppo_random_seed123_3000steps"
python -m train.train --config configs/ppo_config.yaml --algo ppo --hint random --seed 412 --output_dir "runs/ppo_small/ppo_random_seed412_3000steps"
python -m train.train --config configs/ppo_config.yaml --algo ppo --hint random --seed 512 --output_dir "runs/ppo_small/ppo_random_seed512_3000steps"

# REINFORCE experiments - random hints
python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint random --seed 42 --output_dir "runs/reinforce_small/reinforce_random_seed42_2000steps"
python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint random --seed 46 --output_dir "runs/reinforce_small/reinforce_random_seed46_2000steps"
python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint random --seed 123 --output_dir "runs/reinforce_small/reinforce_random_seed123_2000steps"
python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint random --seed 412 --output_dir "runs/reinforce_small/reinforce_random_seed412_2000steps"
python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint random --seed 512 --output_dir "runs/reinforce_small/reinforce_random_seed512_2000steps"

# PPO experiments - expert hints
python -m train.train --config configs/ppo_config.yaml --algo ppo --hint expert --seed 42 --output_dir "runs/ppo_small/ppo_expert_seed42_3000steps"
python -m train.train --config configs/ppo_config.yaml --algo ppo --hint expert --seed 46 --output_dir "runs/ppo_small/ppo_expert_seed46_3000steps"
python -m train.train --config configs/ppo_config.yaml --algo ppo --hint expert --seed 123 --output_dir "runs/ppo_small/ppo_expert_seed123_3000steps"
python -m train.train --config configs/ppo_config.yaml --algo ppo --hint expert --seed 412 --output_dir "runs/ppo_small/ppo_expert_seed412_3000steps"
python -m train.train --config configs/ppo_config.yaml --algo ppo --hint expert --seed 512 --output_dir "runs/ppo_small/ppo_expert_seed512_3000steps"

# REINFORCE experiments - expert hints
python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint expert --seed 42 --output_dir "runs/reinforce_small/reinforce_expert_seed42_2000steps"
python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint expert --seed 46 --output_dir "runs/reinforce_small/reinforce_expert_seed46_2000steps"
python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint expert --seed 123 --output_dir "runs/reinforce_small/reinforce_expert_seed123_2000steps"
python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint expert --seed 412 --output_dir "runs/reinforce_small/reinforce_expert_seed412_2000steps"
python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint expert --seed 512 --output_dir "runs/reinforce_small/reinforce_expert_seed512_2000steps"

# # PPO experiments - LLM hints
# python -m train.train --config configs/ppo_config.yaml --algo ppo --hint llm --seed 42 --output_dir "runs/ppo/ppo_llm_seed42_3000steps"
# python -m train.train --config configs/ppo_config.yaml --algo ppo --hint llm --seed 46 --output_dir "runs/ppo/ppo_llm_seed46_3000steps"
# python -m train.train --config configs/ppo_config.yaml --algo ppo --hint llm --seed 123 --output_dir "runs/ppo/ppo_llm_seed123_3000steps"

# # REINFORCE experiments - LLM hints
# python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint llm --seed 42 --output_dir "runs/reinforce/reinforce_llm_seed42_2000steps"
# python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint llm --seed 46 --output_dir "runs/reinforce/reinforce_llm_seed46_2000steps"
# python -m train.train --config configs/reinforce_config.yaml --algo reinforce --hint llm --seed 123 --output_dir "runs/reinforce/reinforce_llm_seed123_2000steps"
