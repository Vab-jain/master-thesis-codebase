# main.py

import torch
import random
import time
import os

from env import TicTacToeEnv, LLMSuggestionWrapper, RandomSuggestionWrapper
from agent import DQNAgent, RandomAgent
# from u import TicTacToeOracle


from utils import plot_training_results, set_random_seeds, state_to_board
from tqdm import tqdm
# from llm_utils import load_llm_cache
# from llm_utils import get_llm_suggestion, load_llm_cache
# from optim_llm_train import get_llm_suggestion


def train_agent(experiment_config):
    # Unpack experiment configuration
    # global
    experiment_name = experiment_config['experiment_name']
    seed = experiment_config.get('seed', None)
    agent_plays = experiment_config.get('agent_plays', 'random')    # agent plays {first_player, second_player, random}
    agent_policy = experiment_config.get('agent_policy', 'dqn')
    random_suggestion = experiment_config.get('random_suggestion', False)
    env_policy = experiment_config.get('env_policy', None)
    # llm params
    save_llm_cache = experiment_config.get('save_llm_cache', True)
    llm_load_path = experiment_config.get('llm_load_path', None)
    llm_model_id = experiment_config.get('llm_model_id', 'ollama_chat/llama3.2:3b')
    use_GROQ = experiment_config.get('use_GROQ', False)
    llm_cache_path = experiment_config.get('llm_cache_path', None)
    suggestion_one_hot = experiment_config.get('suggestion_one_hot', False)
    # train
    train_num_episodes = experiment_config['train_num_episodes']
    train_llm_use_probability = experiment_config['train_llm_use_probability']      # probability of using LLM suggestion while training
    validation_episodes = experiment_config.get('validation_episodes', [i for i in range(100, 1000, 100)])
    # learning params
    batch_size = experiment_config.get('batch_size', 32)
    learning_rate = experiment_config.get('learning_rate', 1e-4)
    gamma = experiment_config.get('gamma', 1)
    target_update = experiment_config.get('target_update', 100)      # number of steps after which update the target network
    

    # Set random seeds
    set_random_seeds(seed)

    # Create directories to save results
    results_dir = os.path.join('experiments_results/RL_phase', experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    saved_models = os.path.join(results_dir, 'saved_models') 
    os.makedirs(saved_models, exist_ok=True)

    # Initialize environment
    if env_policy=='dqn':
        # load the environment agent
        env_agent_filepath = os.path.join('experiments_results/RL_phase', 'EA-DQN42','saved_models', 'best_tic_tac_toe_agent.pth')
        env_agent = DQNAgent(
                state_size=9, 
                action_size=9, 
                batch_size=batch_size, 
                learning_rate=learning_rate, 
                gamma=gamma, 
                target_update=target_update
            )
        env_agent.load(filepath=env_agent_filepath)
        env_agent.eval()
        env = TicTacToeEnv(env_policy=env_agent)
    else:
        env = TicTacToeEnv()
    if train_llm_use_probability:
        env = LLMSuggestionWrapper(env, 
                                   train_llm_use_probability,
                                   llm_model_id=llm_model_id, 
                                   load_llm_path=llm_load_path,
                                   cache_llm=save_llm_cache,
                                   GROQ=use_GROQ,
                                   llm_cache_path=llm_cache_path,
                                   one_hot_suggestion=suggestion_one_hot
                                   )
    elif random_suggestion:
        env = RandomSuggestionWrapper(env)       
    # Oracle
    # if use_oracle:
    #     oracle = TicTacToeOracle()
    
    # Initialize agent
    if agent_policy == 'dqn':
        agent = DQNAgent(
                state_size=env.observation_space.shape[0], 
                action_size=env.action_space.n.item(), 
                batch_size=batch_size, 
                learning_rate=learning_rate, 
                gamma=gamma, 
                target_update=target_update
            )
    if agent_policy == 'random':
        agent = RandomAgent()
    

    
    # Record training time
    training_start_time = time.time()

    # Training
    win_history = []
    loss_history = []
    draw_history = []
    log_sample_efficiency_win = []
    log_sample_efficiency_loss = []
    log_sample_efficiency_draw = []
    best_wins = 0

    for episode in tqdm(range(1, train_num_episodes + 1), desc="Training Episodes"):
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
            # standard training loop
            # step 1: get RL agnet's action
            action = agent.select_action(state)
            # step 2: take action and get next state and reward
            next_state, reward, done = env.step(action)
            
            if agent_policy == 'dqn':
                # step 3: store the experience in the replay buffer
                agent.memory.push(state, action, reward, next_state, done)
                # step 4: optimize the model after every episode
                agent.optimize_model()
                agent.update_epsilon()
            
            # step 5: update the state and counters
            state = next_state
            agent.steps_done += 1
               
        # Update target network
        if agent_policy == 'dqn':
            if agent.steps_done % agent.target_update == 0:
                agent.update_target_network()
        
        # Update stats
        win_history.append(1 if env.get_winner() == 1 else 0)
        loss_history.append(1 if env.get_winner() == -1 else 0)
        draw_history.append(1 if env.get_winner() == 0 else 0)
        if episode % 100 == 0:
            wins = sum(win_history[-100:])
            losses = sum(loss_history[-100:])
            draws = sum(draw_history[-100:])
            print(f"(Training) Experiment {experiment_name} - Episode {episode}, Win: {wins}, Loss: {losses}, Draw: {draws}")
            if validation_episodes is not None and episode in validation_episodes:
                if agent_policy == 'dqn':
                    agent_filepath = os.path.join(results_dir,'saved_models', f'{episode}_tic_tac_toe_agent.pth')
                    agent.save(agent_filepath)
            # store sample efficiency values
            log_sample_efficiency_win.append(wins)
            log_sample_efficiency_loss.append(losses)
            log_sample_efficiency_draw.append(draws)
            # save model if wins > best_wins
            if wins>best_wins:
                best_wins = wins
                if agent_policy == 'dqn':
                    agent_filepath = os.path.join(results_dir,'saved_models', 'best_tic_tac_toe_agent.pth')
                    agent.save(agent_filepath)
                
            
    # Record training time
    training_end_time = time.time()
    training_time = training_end_time - training_start_time

    # Save the agent
    if agent_policy == 'dqn':
        agent_filepath = os.path.join(results_dir,'saved_models', 'final_tic_tac_toe_agent.pth')
        agent.save(agent_filepath)

    # Plot training results without displaying, save the plot
    plot_filepath = os.path.join(results_dir, 'training_results.png')
    plot_training_results(
        win_history,
        loss_history,
        draw_history,
        agent_plays = agent_plays,  # Agent plays both first and second randomly during the training
        show_plot=False,
        save_path=plot_filepath
    )
    
    # save sample efficiency values
    pth = os.path.join(results_dir, 'sample_efficiency.txt')
    with open(pth, 'w') as f:
        f.write(f'Wins : {log_sample_efficiency_win}')
        f.write(f'Losses : {log_sample_efficiency_loss}')
        f.write(f'Draws : {log_sample_efficiency_draw}')

    train_info = {
        'training_time': training_time,
                  }
    
    return train_info



def eval_agent(experiment_config):
    # Unpack experiment configuration
    # global
    experiment_name = experiment_config['experiment_name']
    random_suggestion = experiment_config.get('random_suggestion', False)
    agent_plays = experiment_config.get('agent_plays', 'random')    # agent plays {first_player, second_player, random}
    agent_policy = experiment_config.get('agent_policy', 'dqn')
    env_policy = experiment_config.get('env_policy', None)
    # eval
    eval_num_games = experiment_config['eval_num_games']
    train_llm_use_probability = experiment_config['train_llm_use_probability']      # probability of using LLM suggestion while training
    eval_llm_use_probability = experiment_config.get('eval_llm_use_probability', train_llm_use_probability)
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
    gamma = experiment_config.get('gamma', 1)
    target_update = experiment_config.get('target_update', 100)
    
    
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
    
    if env_policy=='dqn':
    # load the agent
        env_agent_filepath = os.path.join('experiments_results/RL_phase', 'EA-DQN42','saved_models', 'best_tic_tac_toe_agent.pth')
        env_agent = DQNAgent(
                state_size=9, 
                action_size=9, 
                batch_size=batch_size, 
                learning_rate=learning_rate, 
                gamma=gamma, 
                target_update=target_update
            )
        env_agent.load(filepath=env_agent_filepath)
        env_agent.eval()
        env = TicTacToeEnv(env_policy=env_agent)
    else:
        env = TicTacToeEnv()
    if train_llm_use_probability:
        env = LLMSuggestionWrapper(env, 
                                   eval_llm_use_probability,
                                   llm_model_id=llm_model_id,
                                   load_llm_path=llm_load_path,
                                   cache_llm=save_llm_cache,
                                   GROQ=use_GROQ,
                                   llm_cache_path=llm_cache_path,
                                   one_hot_suggestion=suggestion_one_hot)
    elif random_suggestion:
        env = RandomSuggestionWrapper(env)
        
    # Oracle
    # if use_oracle:
    #     oracle = TicTacToeOracle()
    
    # load the agent
    if agent_policy == 'dqn':
        if eval_agent_load_name:
            agent_filepath = os.path.join('experiments_results/RL_phase', eval_agent_load_name,'saved_models', 'best_tic_tac_toe_agent.pth')
        else:
            agent_filepath = os.path.join(results_dir,'saved_models', 'best_tic_tac_toe_agent.pth')
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


    eval_info = {
        'total_games' : total_games,
        'win' : win,
        'win_percentage' : win_percentage,
        'loss' : loss,
        'loss_percentage' : loss_percentage,
        'draw' : draw,
        'draw_percentage': draw_percentage,
        'evaluation_time' : evaluation_time
    }

    return eval_info
