from env import TicTacToeEnv
import numpy as np
from env import LLMSuggestionWrapper
from agent import DQNAgent, TicTacToeOracle
import os

# PLAY AGAINST ORACLE
# env_agent = TicTacToeOracle()
# env = TicTacToeEnv(env_policy=env_agent)

# PLAY AGAINST DQN AGENT
env_agent_filepath = os.path.join('experiments_results/RL_phase', 'EA-DQN42','saved_models', 'best_tic_tac_toe_agent.pth')
env_agent = DQNAgent(
        state_size=9, 
        action_size=9, 
        batch_size=32, 
        learning_rate=1e-4, 
        gamma=1, 
        target_update=100
    )
env_agent.load(filepath=env_agent_filepath)
env_agent.eval()
env = TicTacToeEnv()

env = LLMSuggestionWrapper(env, 
                           llm_use_probability=1,
                           llm_model_id='llama3-70b-8192',
                           GROQ=True,
                        #    llm_cache_path='LLM_database/E20-GROQ-llama3-70b-ZS.json',
                           )

for episode in range(10):
    print(f'\n###### Begin Episode: {episode+1} ######')
    first_player = np.random.choice([1,-1])
    state = env.reset(first_player)
    env.render()
    done = False
    while not done:
        print(f'LLM Suggestion: {state[-1]}')
        action = int(input('Enter action: '))
        state, reward, done = env.step(action)    
        env.render()
    if reward==1:
        print('You won!')
    elif reward==-1:
        print('You lost!')
    elif reward==-10:
        print('Invalid Action!')
    else:
        print('Draw!')
    # env.render()
    print('#########################################')