import random
import time
import os
from tqdm import tqdm
import csv

from env import TicTacToeEnv, LLMSuggestionWrapper, RandomSuggestionWrapper
from agent import DQNAgent, RandomAgent
from config import update_config

arr = [i for i in range(100,1000,100)] 
arr.extend(['final'])
# arr.extend(['best', 'final'])

def val_agent(experiment_config):
    # Unpack experiment configuration
    # global
    experiment_name = experiment_config['experiment_name']
    random_suggestion = experiment_config.get('random_suggestion', False)
    agent_plays = experiment_config.get('agent_plays', 'random')    # agent plays {first_player, second_player, random}
    agent_policy = experiment_config.get('agent_policy', 'dqn')
    # eval
    eval_num_games = experiment_config['eval_num_games']
    train_llm_use_probability = experiment_config['train_llm_use_probability']      # probability of using LLM suggestion while training
    eval_llm_use_probability = experiment_config.get('eval_llm_use_probability',train_llm_use_probability)
    eval_agent_load_name = experiment_config.get('eval_agent_load_name', None)
    # llm params
    save_llm_cache = experiment_config.get('save_llm_cache', True)
    llm_load_path = experiment_config.get('llm_load_path', None)
    llm_model_id = experiment_config.get('llm_model_id', 'ollama_chat/llama3.2:3b')
    use_GROQ = experiment_config.get('use_GROQ', False)
    llm_cache_path = experiment_config.get('llm_cache_path', None)
    suggestion_one_hot = experiment_config.get('suggestion_one_hot', False)
    # learning params
    batch_size = experiment_config.get('batch_size', 32)
    learning_rate = experiment_config.get('learning_rate', 1e-4)
    gamma = experiment_config.get('gamma', 0.99)
    target_update = experiment_config.get('target_update', 10)
    
    
    # Record evaluation time
    evaluation_start_time = time.time()
    
    # Create directories to load results
    results_dir = os.path.join('experiments_results/RL_phase', experiment_name)

    # Check if the directory exists
    if os.path.exists(results_dir) and os.path.isdir(results_dir):
        print(f"Loading Experiment...")
    else:
        print(f"Experiment '{experiment_name}' does not exist.")
        return
    
    # Initialize environment
    env = TicTacToeEnv()
    if train_llm_use_probability:
        env = LLMSuggestionWrapper(env, 
                                   eval_llm_use_probability,
                                   llm_model_id=llm_model_id,
                                   load_llm_path=llm_load_path,
                                   cache_llm=save_llm_cache,
                                   GROQ=use_GROQ,
                                   llm_cache_path=llm_cache_path,
                                   one_hot_suggestion=suggestion_one_hot,
                                   )
    elif random_suggestion:
        env = RandomSuggestionWrapper(env)
        
    
    for name in arr:    
        # load the agent
        if agent_policy == 'dqn':
            if eval_agent_load_name:
                agent_filepath = os.path.join('experiments_results/RL_phase', eval_agent_load_name,'saved_models', f'{name}_tic_tac_toe_agent.pth')
            else:
                agent_filepath = os.path.join(results_dir,'saved_models', f'{name}_tic_tac_toe_agent.pth')
            agent = DQNAgent(
                    state_size=env.observation_space.shape[0], 
                    action_size=env.action_space.n.item(), 
                    batch_size=batch_size, 
                    learning_rate=learning_rate, 
                    gamma=gamma, 
                    target_update=target_update
                )
            agent.load(filepath=agent_filepath)
            agent.eval()
        if agent_policy == 'random':
            agent = RandomAgent()

        
        # Evaluation
        win, loss, draw = 0, 0, 0
        for _ in tqdm(range(eval_num_games), desc="Evaluation Games"):
            # Determine who plays first based on agent_plays
            if agent_plays == 'first_player':
                first_player = 1  # RL agent plays first
            elif agent_plays == 'second_player':
                first_player = -1  # Envoronment agent plays first
            else:  # Randomly assign who plays first
                first_player = random.choice([1, -1])

            # reset envrioment at the beginning of the episode
            state = env.reset(first_player=first_player)
            done = False
            
            while not done:
                action = agent.select_action(state)  # No exploration during evaluation
                state, reward, done = env.step(action)
            
            # Update stats
            if env.get_winner() == 1:
                win += 1
            elif env.get_winner() == -1:
                loss +=1
            elif env.get_winner() == 0:
                draw +=1

        # Record evaluation time
        evaluation_end_time = time.time()
        evaluation_time = evaluation_end_time - evaluation_start_time

        total_games = win + loss + draw
        win_percentage = (win / total_games) * 100
        loss_percentage = (loss / total_games) * 100
        draw_percentage = (draw / total_games) * 100
        avg_return = (win - loss) / total_games


        eval_info = {
            'checkpoint' : name,
            'total_games' : total_games,
            'win' : win,
            'win_percentage' : win_percentage,
            'loss' : loss,
            'loss_percentage' : loss_percentage,
            'draw' : draw,
            'draw_percentage': draw_percentage,
            'average_return' : avg_return,
        }

        print(eval_info)
        if eval_agent_load_name:
            save_filepath = os.path.join('experiments_results/RL_phase', eval_agent_load_name, 'sample_efficiency.json')
        else:
            save_filepath = os.path.join(results_dir, 'sample_efficiency.csv')

        try:
            with open(save_filepath, 'r') as f:
                # If file exists, don't write headers
                pass
        except FileNotFoundError:
            # File doesn't exist, so write headers
            with open(save_filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=eval_info.keys())
                writer.writeheader()
        with open(save_filepath,'a') as f:
            writer = csv.DictWriter(f, fieldnames=eval_info.keys())
            writer.writerow(eval_info)


experiment_config = [
    {
    #### global
    'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT',
    'seed': 42,
    # 'env_policy' : 'dqn',
    #### llm
    # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
    'llm_model_id' : 'llama3-70b-8192',
    'use_GROQ' : True,
    'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
    'suggestion_one_hot' : False,
    #### train
    'train_num_episodes': 0,
    'train_llm_use_probability': 0.5,
    #### eval
    'eval_num_games': 100,
    'eval_llm_use_probability' : 1,
    ######## update config params
    'board_representation' : '1D',
    'cross_representation' : '1/-1',
    'task_description' : 'td_og',
    'prompting_method' : 'CoT',
    },
    {
    #### global
    'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT_seed50',
    'seed': 50,
    # 'env_policy' : 'dqn',
    #### llm
    # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
    'llm_model_id' : 'llama3-70b-8192',
    'use_GROQ' : True,
    'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
    'suggestion_one_hot' : False,
    #### train
    'train_num_episodes': 0,
    'train_llm_use_probability': 0.5,
    #### eval
    'eval_num_games': 100,
    'eval_llm_use_probability' : 1,
    ######## update config params
    'board_representation' : '1D',
    'cross_representation' : '1/-1',
    'task_description' : 'td_og',
    'prompting_method' : 'CoT',
    },
    {
    #### global
    'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT_seed70',
    'seed': 70,
    # 'env_policy' : 'dqn',
    #### llm
    # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
    'llm_model_id' : 'llama3-70b-8192',
    'use_GROQ' : True,
    'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
    'suggestion_one_hot' : False,
    #### train
    'train_num_episodes': 0,
    'train_llm_use_probability': 0.5,
    #### eval
    'eval_num_games': 100,
    'eval_llm_use_probability' : 1,
    ######## update config params
    'board_representation' : '1D',
    'cross_representation' : '1/-1',
    'task_description' : 'td_og',
    'prompting_method' : 'CoT',
    },
        
        ]
for experiment in experiment_config:
    print(f'Benchmarking: {experiment["experiment_name"]}')
    update_config(experiment['board_representation'], 
                      experiment['cross_representation'], 
                      experiment['task_description'], 
                      experiment['prompting_method'],)
    val_agent(experiment)