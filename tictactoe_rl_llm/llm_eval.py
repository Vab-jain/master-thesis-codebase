import os
import tqdm
import numpy as np
import time
import json

from utils import set_random_seeds

from env import TicTacToeEnv, LLMSuggestionWrapper

def llm_play_eval(
            experiment_config, 
            ):
    # Unpack experiment configuration
    experiment_name = experiment_config['experiment_name']
    seed = experiment_config.get('seed', None)
    llm_model_id = experiment_config['llm_model_id']
    load_llm_path = experiment_config.get('load_llm_path', None)
    eval_num_games = experiment_config.get('eval_num_games', 100)
    cache_llm_database = experiment_config.get('cache_llm_database', False)
    use_GROQ = experiment_config.get('use_GROQ', False)
    llm_cache_path = experiment_config.get('llm_cache_path', None)
    
    env = TicTacToeEnv()
    env = LLMSuggestionWrapper(env, 
                            llm_use_probability=1, 
                            llm_model_id=llm_model_id,
                            load_llm_path=load_llm_path,
                            GROQ=use_GROQ,
                            llm_cache_path=llm_cache_path,
                            )

    '''
        * Test the LLM suggestion accuracy *
        - Let LLM play the game
        - Count the number of wins, losses, draws, and invalid actions
    '''
    win_ctr = 0
    loss_ctr = 0
    draw_ctr = 0
    invalid_ctr = 0
    suggestion_ctr = 0
    
    set_random_seeds(seed)
    
    # cache json file
    # Create an empty dictionary to store state-action pairs
    state_action_dict = {}

    # Function to save the state-action pair
    def save_state_action_pair(state, action):
        state_key = json.dumps(state)
        state_action_dict[state_key] = int(action)  # Convert state to a tuple for hashability


    for episode in tqdm.tqdm(range(eval_num_games)):
        first_player = np.random.choice([1,-1])
        state = env.reset(first_player)
        done = False
        while not done:
            if use_GROQ:
                time.sleep(2)
            suggestion_ctr += 1
            action = state[-1]
            if action==-1:
                reward = -10
                break
            next_state, reward, done = env.step(action)
            # cache llm output
            if reward==-10:
                pass
            elif json.dumps(state[:9].tolist()) in state_action_dict:
                if reward==1:
                    save_state_action_pair(state[:9].tolist(), action)
            else:
                save_state_action_pair(state[:9].tolist(), action)
            
            state = next_state
            
        if reward==1:
            win_ctr += 1
        elif reward==-1:
            loss_ctr += 1
        elif reward==-10:
            invalid_ctr += 1
        elif reward==0:
            draw_ctr += 1
        else:
            exit('Unknown reward')
    
    print('Saving Results...')
    RESULTS_PATH = os.path.join("experiments_results", "LLM_phase", f"{experiment_name}.txt")
    with open(RESULTS_PATH, 'a') as file:
        file.write('Experiment Details:\n')
        file.write('-------------------\n')
        file.write(f'Experiment Name: {experiment_name}\n')
        file.write(f'Seed: {seed}\n')
        file.write(f'LLM: {llm_model_id}\n')
        file.write('LLM Agent Plays: Random\n')
        file.write(f'Number of Games: {eval_num_games}\n')
        file.write('-------------------\n\n')

        file.write('LLM Agent Accuracy:\n')
        file.write(f'Win: {win_ctr}, Loss: {loss_ctr}, Draw: {draw_ctr}, Invalid: {invalid_ctr}, Invalid (%): {100*invalid_ctr/suggestion_ctr}, Total Suggestions: {suggestion_ctr}\n')
        file.write('#####################\n\n')
    print(f'Results saved successfully at: {RESULTS_PATH}')
    
    if cache_llm_database:
        with open(f'LLM_database/{experiment_name}.json', 'w') as f:
            json.dump(state_action_dict, f, indent=4)