from llm_eval import llm_play_eval
from config import update_config

def main():
    experiments = [
        {
            'experiment_name': 'E20-llama3-70b-FS1_CoT',
            'seed': 42,
            # 'llm_model_id' : 'ollama_chat/llama2:13b',
            'llm_model_id' : 'llama3-70b-8192',
            'load_llm_path' : 'saved_optim_llm/optim_llama3-70b__dqn2__1D_int__metricV2__top1__AA_FS1/optim_llm.pkl',
            'eval_num_games' : 100,
            'use_GROQ' : True,
            ######## update config params
            'board_representation' : '1D',
            'cross_representation' : '1/-1',
            'task_description' : 'td_og',
            'prompting_method' : 'CoT',
        },
        {
            'experiment_name': 'E20-llama3-70b-FS3_CoT',
            'seed': 42,
            # 'llm_model_id' : 'ollama_chat/llama2:13b',
            'llm_model_id' : 'llama3-70b-8192',
            'load_llm_path' : 'saved_optim_llm/optim_llama3-70b__dqn2__1D_int__metricV2__top1__AA_FS3/optim_llm.pkl',
            'eval_num_games' : 100,
            'use_GROQ' : True,
            ######## update config params
            'board_representation' : '1D',
            'cross_representation' : '1/-1',
            'task_description' : 'td_og',
            'prompting_method' : 'CoT',
        },
        {
            'experiment_name': 'E20-llama3-70b-FS5_CoT',
            'seed': 42,
            # 'llm_model_id' : 'ollama_chat/llama2:13b',
            'llm_model_id' : 'llama3-70b-8192',
            'load_llm_path' : 'saved_optim_llm/optim_llama3-70b__dqn2__1D_int__metricV2__top1__AA_FS5/optim_llm.pkl',
            'eval_num_games' : 100,
            'use_GROQ' : True,
            ######## update config params
            'board_representation' : '1D',
            'cross_representation' : '1/-1',
            'task_description' : 'td_og',
            'prompting_method' : 'CoT',
        },
        
        
    ]
    
    for experiment_config in experiments:
        update_config(experiment_config['board_representation'], 
                      experiment_config['cross_representation'], 
                      experiment_config['task_description'], 
                      experiment_config['prompting_method'],
                      )
        llm_play_eval(experiment_config)

if __name__ == '__main__':
    main()