import os
import config

def log_experiment(experiment_config, train_info=None, eval_info=None):
    # Unpack Experiment Configuration
    # global
    experiment_name = experiment_config['experiment_name']
    seed = experiment_config.get('seed')
    agent_plays = experiment_config.get('agent_plays', 'random')    # agent plays {first_player, second_player, random}
    agent_policy = experiment_config.get('agent_policy', 'dqn')
    random_suggestion = experiment_config.get('random_suggestion', False)

    # Training Parameters
    num_episodes = experiment_config['train_num_episodes']
    llm_use_probability = experiment_config['train_llm_use_probability']

    # LLM Parameters
    save_llm_cache = experiment_config.get('save_llm_cache', True)
    llm_load_path = experiment_config.get('llm_load_path', None)
    llm_model_id = experiment_config.get('llm_model_id', 'ollama_chat/llama3.2:3b')
    use_GROQ = experiment_config.get('use_GROQ', False)
    llm_cache_path = experiment_config.get('llm_cache_path', None)
    suggestion_one_hot = experiment_config.get('suggestion_one_hot', False)

    # Learning Parameters
    batch_size = experiment_config.get('batch_size', 32)
    learning_rate = experiment_config.get('learning_rate', 1e-4)
    gamma = experiment_config.get('gamma', 0.99)
    target_update = experiment_config.get('target_update', 30)

    # Evaluation Parameters
    eval_llm_use_probability = experiment_config.get('eval_llm_use_probability', llm_use_probability)
    eval_num_games = experiment_config['eval_num_games']
    eval_agent_load_name = experiment_config.get('eval_agent_load_name')

    # Capture Training Info
    training_time = train_info['training_time'] if train_info else 0

    # Capture Evaluation Info
    total_games = eval_info['total_games']
    win = eval_info['win']
    win_percentage = eval_info['win_percentage']
    loss = eval_info['loss']
    loss_percentage = eval_info['loss_percentage']
    draw = eval_info['draw']
    draw_percentage = eval_info['draw_percentage']
    evaluation_time = eval_info['evaluation_time']

    # Create results directory
    results_dir = os.path.join('experiments_results/RL_phase', experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    # Log results
    log_filepath = os.path.join(results_dir, 'experiment_log.txt')
    with open(log_filepath, 'w') as f:
        f.write(f"### {experiment_name} ###\n\n")
        f.write("Config\n")
        f.write('-------------------\n')
        f.write(f"Seed: {seed}\n")
        f.write(f"Agent First Strategy: {agent_plays}\n")
        f.write(f"Agent Policy: {agent_policy}\n")
        f.write(f"Random Suggestion: {random_suggestion}\n\n")
        
        f.write(f"Board Representation: {config.board_representation}\n")
        f.write(f"Cross Representation: {config.cross_representation}\n")
        f.write(f"DSPY Signature: {config.dspy_signature}\n")
        f.write(f"Prompting Method: {config.prompting_method}\n")
        f.write(f"Task Description: {experiment_config['task_description']}\n")
        

        f.write("LLM Config\n")
        f.write('-------------------\n')
        f.write(f"Save LLM Cache: {save_llm_cache}\n")
        f.write(f"LLM Load Path: {llm_load_path}\n")
        f.write(f"LLM Model ID: {llm_model_id}\n")
        f.write(f"Use GROQ: {use_GROQ}\n")
        f.write(f"LLM Cache Path: {llm_cache_path}\n")
        f.write(f"Suggestion One Hot: {suggestion_one_hot}\n\n")

        f.write("Learning Config\n")
        f.write('-------------------\n')
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Gamma: {gamma}\n")
        f.write(f"Target Update Steps: {target_update}\n\n")

        f.write("Training\n")
        f.write('-------------------\n')
        f.write(f"Training Episodes: {num_episodes}\n")
        f.write(f"LLM Use Probability (Training): {llm_use_probability}\n")
        f.write("Training Time: {:.2f} seconds\n".format(training_time))
        
        f.write("\nEvaluation\n")
        f.write('-------------------\n')
        if eval_agent_load_name:
            f.write(f"Agent Evaluation Policy: {eval_agent_load_name}")
        f.write(f"Total Games: {total_games}\n")
        f.write(f"LLM Use Probability (Evaluation): {eval_llm_use_probability}\n")
        f.write(f"Win: {win} ({win_percentage:.2f}%)\n")
        f.write(f"Loss: {loss} ({loss_percentage:.2f}%)\n")
        f.write(f"Draw: {draw} ({draw_percentage:.2f}%)\n")
        f.write(f"Average Return: {win - loss / total_games}\n")
        f.write(f"Evaluation Time: {evaluation_time:.2f} seconds\n\n")


    # Print Summary
    print(f"\nExperiment {experiment_name} completed.")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Evaluation Time: {evaluation_time:.2f} seconds")
    print(f"Win: {win} ({win_percentage:.2f}%), Loss: {loss} ({loss_percentage:.2f}%), Draw: {draw} ({draw_percentage:.2f}%)")
    print(f"Results saved in: {results_dir}")
