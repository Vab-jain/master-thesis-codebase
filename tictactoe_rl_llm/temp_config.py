#####################################################################
        ################## DQN TRAINING EXPERIMENTS ########################
        #####################################################################
        # {
        #     #### global
        #     'experiment_name': 'EA-DQN50',
        #     'seed': 50,
        #     # 'env_policy' : 'dqn',
        #     #### llm
        #     # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
        #     # 'llm_model_id' : 'llama3-70b-8192',
        #     # 'use_GROQ' : True,
        #     # 'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
        #     # 'suggestion_one_hot' : False,
        #     #### train
        #     'train_num_episodes': 1000,
        #     'train_llm_use_probability': 0,
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
        #     'experiment_name': 'EA-DQN70',
        #     'seed': 70,
        #     # 'env_policy' : 'dqn',
        #     #### llm
        #     # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
        #     # 'llm_model_id' : 'llama3-70b-8192',
        #     # 'use_GROQ' : True,
        #     # 'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
        #     # 'suggestion_one_hot' : False,
        #     #### train
        #     'train_num_episodes': 1000,
        #     'train_llm_use_probability': 0,
        #     #### eval
        #     'eval_num_games': 100,
        #     'eval_llm_use_probability' : 0,
        #     ######## update config params
        #     'board_representation' : '1D',
        #     'cross_representation' : '1/-1',
        #     'task_description' : 'td_og',
        #     'prompting_method' : 'CoT',
        # },
        ###################################################################


# ######################################################################
# ################ SINGLE INT ENCODING LLM-suggestion ##################
# ######################################################################
# {
#     #### global
#     'experiment_name': 'EF-llama3-70b-1D_int-CoT_seed50',
#     'seed': 50,
#     # 'env_policy' : 'dqn',
#     #### llm
#     # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
#     'llm_model_id' : 'llama3-70b-8192',
#     'use_GROQ' : True,
#     'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
#     'suggestion_one_hot' : False,
#     #### train
#     'train_num_episodes': 1000,
#     'train_llm_use_probability': 1,
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
#     'experiment_name': 'EF-llama3-70b-1D_int-CoT_seed70',
#     'seed': 70,
#     # 'env_policy' : 'dqn',
#     #### llm
#     # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
#     'llm_model_id' : 'llama3-70b-8192',
#     'use_GROQ' : True,
#     'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
#     'suggestion_one_hot' : False,
#     #### train
#     'train_num_episodes': 1000,
#     'train_llm_use_probability': 1,
#     #### eval
#     'eval_num_games': 100,
#     'eval_llm_use_probability' : 1,
#     ######## update config params
#     'board_representation' : '1D',
#     'cross_representation' : '1/-1',
#     'task_description' : 'td_og',
#     'prompting_method' : 'CoT',
# },
# ###########################################################################

# ###########################################################################
# ######################################################################
# ################ One-hot ENCODING LLM-suggestion ##################
# ######################################################################
# {
#     #### global
#     'experiment_name': 'EF-llama3-70b-1D_int-CoT-OH_seed50',
#     'seed': 50,
#     # 'env_policy' : 'dqn',
#     #### llm
#     # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
#     'llm_model_id' : 'llama3-70b-8192',
#     'use_GROQ' : True,
#     'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
#     'suggestion_one_hot' : True,
#     #### train
#     'train_num_episodes': 1000,
#     'train_llm_use_probability': 1,
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
#     'experiment_name': 'EF-llama3-70b-1D_int-CoT-OH_seed70',
#     'seed': 70,
#     # 'env_policy' : 'dqn',
#     #### llm
#     # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
#     'llm_model_id' : 'llama3-70b-8192',
#     'use_GROQ' : True,
#     'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
#     'suggestion_one_hot' : True,
#     #### train
#     'train_num_episodes': 1000,
#     'train_llm_use_probability': 1,
#     #### eval
#     'eval_num_games': 100,
#     'eval_llm_use_probability' : 1,
#     ######## update config params
#     'board_representation' : '1D',
#     'cross_representation' : '1/-1',
#     'task_description' : 'td_og',
#     'prompting_method' : 'CoT',
# },
# ###########################################################################
# ######################################################################
# ################ SINGLE INT ENCODING LLM-suggestion 50% ##############
# ######################################################################
# {
    # #### global
    # 'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT',
    # 'seed': 42,
    # # 'env_policy' : 'dqn',
    # #### llm
    # # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
    # 'llm_model_id' : 'llama3-70b-8192',
    # 'use_GROQ' : True,
    # 'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
    # 'suggestion_one_hot' : False,
    # #### train
    # 'train_num_episodes': 1000,
    # 'train_llm_use_probability': 0.5,
    # #### eval
    # 'eval_num_games': 100,
    # 'eval_llm_use_probability' : 1,
    # ######## update config params
    # 'board_representation' : '1D',
    # 'cross_representation' : '1/-1',
    # 'task_description' : 'td_og',
    # 'prompting_method' : 'CoT',
    # },
    # {
    # #### global
    # 'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT_seed50',
    # 'seed': 50,
    # # 'env_policy' : 'dqn',
    # #### llm
    # # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
    # 'llm_model_id' : 'llama3-70b-8192',
    # 'use_GROQ' : True,
    # 'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
    # 'suggestion_one_hot' : False,
    # #### train
    # 'train_num_episodes': 1000,
    # 'train_llm_use_probability': 0.5,
    # #### eval
    # 'eval_num_games': 100,
    # 'eval_llm_use_probability' : 1,
    # ######## update config params
    # 'board_representation' : '1D',
    # 'cross_representation' : '1/-1',
    # 'task_description' : 'td_og',
    # 'prompting_method' : 'CoT',
    # },
    # {
    # #### global
    # 'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT_seed70',
    # 'seed': 70,
    # # 'env_policy' : 'dqn',
    # #### llm
    # # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
    # 'llm_model_id' : 'llama3-70b-8192',
    # 'use_GROQ' : True,
    # 'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
    # 'suggestion_one_hot' : False,
    # #### train
    # 'train_num_episodes': 1000,
    # 'train_llm_use_probability': 0.5,
    # #### eval
    # 'eval_num_games': 100,
    # 'eval_llm_use_probability' : 1,
    # ######## update config params
    # 'board_representation' : '1D',
    # 'cross_representation' : '1/-1',
    # 'task_description' : 'td_og',
    # 'prompting_method' : 'CoT',
    # },   
        
        
        
# ######################################################################
# ################ One-hot ENCODING LLM-suggestion 50% ##############
# ######################################################################
#################################################################
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
experiments = [
        {
            #### global
            'experiment_name': 'EF-llama3-70b-1D_int-CoT',
            'seed': 42,
            # 'env_policy' : 'dqn',
            #### llm
            # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
            'llm_model_id' : 'llama3-70b-8192',
            'use_GROQ' : True,
            'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
            'suggestion_one_hot' : False,
            #### train
            'train_num_episodes': 1000,
            'train_llm_use_probability': 1,
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
            'experiment_name': 'EF-llama3-70b-1D_int-CoT-OH',
            'seed': 42,
            # 'env_policy' : 'dqn',
            #### llm
            # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
            'llm_model_id' : 'llama3-70b-8192',
            'use_GROQ' : True,
            'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
            'suggestion_one_hot' : True,
            #### train
            'train_num_episodes': 1000,
            'train_llm_use_probability': 1,
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
            'train_num_episodes': 1000,
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
            'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT-OH',
            'seed': 42,
            # 'env_policy' : 'dqn',
            #### llm
            # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
            'llm_model_id' : 'llama3-70b-8192',
            'use_GROQ' : True,
            'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
            'suggestion_one_hot' : True,
            #### train
            'train_num_episodes': 1000,
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
            'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT-noEval',
            'seed': 42,
            # 'env_policy' : 'dqn',
            #### llm
            # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
            'llm_model_id' : 'llama3-70b-8192',
            'use_GROQ' : True,
            'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
            'suggestion_one_hot' : False,
            #### train
            'train_num_episodes': 1000,
            'train_llm_use_probability': 0.5,
            #### eval
            'eval_num_games': 100,
            'eval_llm_use_probability' : 0,
            ######## update config params
            'board_representation' : '1D',
            'cross_representation' : '1/-1',
            'task_description' : 'td_og',
            'prompting_method' : 'CoT',
        },
        {
            #### global
            'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT-OH-noEval',
            'seed': 42,
            # 'env_policy' : 'dqn',
            #### llm
            # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
            'llm_model_id' : 'llama3-70b-8192',
            'use_GROQ' : True,
            'llm_cache_path' : 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
            'suggestion_one_hot' : True,
            #### train
            'train_num_episodes': 1000,
            'train_llm_use_probability': 0.5,
            #### eval
            'eval_num_games': 100,
            'eval_llm_use_probability' : 0,
            ######## update config params
            'board_representation' : '1D',
            'cross_representation' : '1/-1',
            'task_description' : 'td_og',
            'prompting_method' : 'CoT',
        },
        ]
    
new_exp = [
    # Seed 123 versions
    {
        'experiment_name': 'EF-llama3-70b-1D_int-CoT-seed123',
        'seed': 123,
        'llm_model_id': 'llama3-70b-8192',
        'use_GROQ': True,
        'llm_cache_path': 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
        'suggestion_one_hot': False,
        'train_num_episodes': 1000,
        'train_llm_use_probability': 1,
        'eval_num_games': 100,
        'eval_llm_use_probability': 1,
        'board_representation': '1D',
        'cross_representation': '1/-1',
        'task_description': 'td_og',
        'prompting_method': 'CoT',
    },
    {
        'experiment_name': 'EF-llama3-70b-1D_int-CoT-OH-seed123',
        'seed': 123,
        'llm_model_id': 'llama3-70b-8192',
        'use_GROQ': True,
        'llm_cache_path': 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
        'suggestion_one_hot': True,
        'train_num_episodes': 1000,
        'train_llm_use_probability': 1,
        'eval_num_games': 100,
        'eval_llm_use_probability': 1,
        'board_representation': '1D',
        'cross_representation': '1/-1',
        'task_description': 'td_og',
        'prompting_method': 'CoT',
    },
    {
        'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT-seed123',
        'seed': 123,
        'llm_model_id': 'llama3-70b-8192',
        'use_GROQ': True,
        'llm_cache_path': 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
        'suggestion_one_hot': False,
        'train_num_episodes': 1000,
        'train_llm_use_probability': 0.5,
        'eval_num_games': 100,
        'eval_llm_use_probability': 1,
        'board_representation': '1D',
        'cross_representation': '1/-1',
        'task_description': 'td_og',
        'prompting_method': 'CoT',
    },
    {
        'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT-OH-seed123',
        'seed': 123,
        'llm_model_id': 'llama3-70b-8192',
        'use_GROQ': True,
        'llm_cache_path': 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
        'suggestion_one_hot': True,
        'train_num_episodes': 1000,
        'train_llm_use_probability': 0.5,
        'eval_num_games': 100,
        'eval_llm_use_probability': 1,
        'board_representation': '1D',
        'cross_representation': '1/-1',
        'task_description': 'td_og',
        'prompting_method': 'CoT',
    },
    {
        'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT-noEval-seed123',
        'seed': 123,
        'llm_model_id': 'llama3-70b-8192',
        'use_GROQ': True,
        'llm_cache_path': 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
        'suggestion_one_hot': False,
        'train_num_episodes': 1000,
        'train_llm_use_probability': 0.5,
        'eval_num_games': 100,
        'eval_llm_use_probability': 0,
        'board_representation': '1D',
        'cross_representation': '1/-1',
        'task_description': 'td_og',
        'prompting_method': 'CoT',
    },
    {
        'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT-OH-noEval-seed123',
        'seed': 123,
        'llm_model_id': 'llama3-70b-8192',
        'use_GROQ': True,
        'llm_cache_path': 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
        'suggestion_one_hot': True,
        'train_num_episodes': 1000,
        'train_llm_use_probability': 0.5,
        'eval_num_games': 100,
        'eval_llm_use_probability': 0,
        'board_representation': '1D',
        'cross_representation': '1/-1',
        'task_description': 'td_og',
        'prompting_method': 'CoT',
    },
    # Seed 999 versions
    {
        'experiment_name': 'EF-llama3-70b-1D_int-CoT-seed999',
        'seed': 999,
        'llm_model_id': 'llama3-70b-8192',
        'use_GROQ': True,
        'llm_cache_path': 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
        'suggestion_one_hot': False,
        'train_num_episodes': 1000,
        'train_llm_use_probability': 1,
        'eval_num_games': 100,
        'eval_llm_use_probability': 1,
        'board_representation': '1D',
        'cross_representation': '1/-1',
        'task_description': 'td_og',
        'prompting_method': 'CoT',
    },
    {
        'experiment_name': 'EF-llama3-70b-1D_int-CoT-OH-seed999',
        'seed': 999,
        'llm_model_id': 'llama3-70b-8192',
        'use_GROQ': True,
        'llm_cache_path': 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
        'suggestion_one_hot': True,
        'train_num_episodes': 1000,
        'train_llm_use_probability': 1,
        'eval_num_games': 100,
        'eval_llm_use_probability': 1,
        'board_representation': '1D',
        'cross_representation': '1/-1',
        'task_description': 'td_og',
        'prompting_method': 'CoT',
    },
    {
        'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT-seed999',
        'seed': 999,
        'llm_model_id': 'llama3-70b-8192',
        'use_GROQ': True,
        'llm_cache_path': 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
        'suggestion_one_hot': False,
        'train_num_episodes': 1000,
        'train_llm_use_probability': 0.5,
        'eval_num_games': 100,
        'eval_llm_use_probability': 1,
        'board_representation': '1D',
        'cross_representation': '1/-1',
        'task_description': 'td_og',
        'prompting_method': 'CoT',
    },
    {
        'experiment_name': 'EF-llama3-70b-1D_int-llm50-CoT-OH-seed999',
        'seed': 999,
        'llm_model_id': 'llama3-70b-8192',
        'use_GROQ': True,
        'llm_cache_path': 'LLM_database/E20-GROQ-llama3-70b-ZS.json',
        'suggestion_one_hot': True,
        'train_num_episodes': 1000,
        'train_llm_use_probability': 0.5,
        'eval_num_games': 100,
        'eval_llm_use_probability': 1,
        'board_representation': '1D',
        'cross_representation': '1/-1',
        'task_description': 'td_og',
        'prompting_method': 'CoT',
    },
]

experiments.extend(new_exp)


print(len(experiments))
print(len(new_exp))