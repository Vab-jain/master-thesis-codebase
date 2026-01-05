# experiments.py
from train_eval_handlers import train_agent, eval_agent
from utils import log_experiment
from config import update_config

def main():
    experiments = [
        
        # {
        #     #### global
        #     'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT-OH',
        #     'seed': 42,
        #     # 'env_policy' : 'dqn',
        #     #### llm
        #     # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
        #     'llm_model_id' : 'llama3-70b-8192',
        #     'use_GROQ' : True,
        #     'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
        #     'suggestion_one_hot' : True,
        #     #### train
        #     'train_num_episodes': 1000,
        #     'train_llm_use_probability': 0.5,
        #     #### eval
        #     'eval_num_games': 100,
        #     'eval_llm_use_probability' : 1,
        #     ######## update config params
        #     'board_representation' : '1D',
        #     'cross_representation' : '1/-1',
        #     'task_description' : 'td_og',
        #     'prompting_method' : 'CoT',
        # },
        # {
        #     #### global
        #     'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT-noEval',
        #     'seed': 42,
        #     # 'env_policy' : 'dqn',
        #     #### llm
        #     # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
        #     'llm_model_id' : 'llama3-70b-8192',
        #     'use_GROQ' : True,
        #     'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
        #     'suggestion_one_hot' : False,
        #     #### train
        #     'train_num_episodes': 1000,
        #     'train_llm_use_probability': 0.5,
        #     #### eval
        #     'eval_num_games': 100,
        #     'eval_llm_use_probability' : 0,
        #     ######## update config params
        #     'board_representation' : '1D',
        #     'cross_representation' : '1/-1',
        #     'task_description' : 'td_og',
        #     'prompting_method' : 'CoT',
        # },
        # {
        #     #### global
        #     'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT-OH-noEval',
        #     'seed': 42,
        #     # 'env_policy' : 'dqn',
        #     #### llm
        #     # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
        #     'llm_model_id' : 'llama3-70b-8192',
        #     'use_GROQ' : True,
        #     'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
        #     'suggestion_one_hot' : True,
        #     #### train
        #     'train_num_episodes': 1000,
        #     'train_llm_use_probability': 0.5,
        #     #### eval
        #     'eval_num_games': 100,
        #     'eval_llm_use_probability' : 0,
        #     ######## update config params
        #     'board_representation' : '1D',
        #     'cross_representation' : '1/-1',
        #     'task_description' : 'td_og',
        #     'prompting_method' : 'CoT',
        # },
        ]
    for experiment_config in experiments:
        print(f"Starting experiment: {experiment_config['experiment_name']}")
        update_config(experiment_config['board_representation'], 
                      experiment_config['cross_representation'], 
                      experiment_config['task_description'], 
                      experiment_config['prompting_method'],)
        train_info = train_agent(experiment_config)
        eval_info = eval_agent(experiment_config)
        log_experiment(experiment_config, train_info, eval_info)

if __name__ == '__main__':
    main()


