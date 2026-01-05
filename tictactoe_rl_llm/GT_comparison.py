import dspy
from dspy.evaluate import Evaluate
import random
import os

import config
import utils
from utils import GenerateAction, TaskContext



def GT_comparison(
                llm_model_id,
                GT_dataset,
                board_representation=config.board_representation,
                cross_representataion=config.cross_representation,
                GROQ=False
                ):
    '''
    Function to optimize LLM
    
    Args:
        llm_model_id  : 
        GT_dataset  :
        board_representation : (default) '1D',
        cross_representataion : (default) '1/-1'
    
    Return:
        llm_accuracy : 
    '''
    
    # configure LLM
    if GROQ:
        lm = dspy.LM(llm_model_id, api_base='https://api.groq.com/openai/v1', api_key=config.GROQ_API_KEY, cache=False)
    else:
        lm = dspy.LM(llm_model_id, api_base='http://localhost:11434', cache=False)
    dspy.configure(lm=lm)
    llm_agent = dspy.ChainOfThought(GenerateAction)

    print('LLM Configured!')

    # Create the Task-Context
    taskcontext = TaskContext(board_representation,cross_representataion,)

    # load the dataset
    df = utils.read_txt_to_dataframe(GT_dataset)
    trainset, devset = [], []
    # train_size = int(0.2 * len(df))  # First 20% for trainset
    train_size = int(len(df)-100)  # All except last 100 for trainset
    for i, row in df.iterrows():
        # Create the example using the board_str and action
        example = dspy.Example(context=taskcontext.context, current_state=row["board_str"], available_actions=row["available_actions"], answer=row["actions"]).with_inputs('context','current_state', 'available_actions')
        if i<train_size:
            trainset.append(example)
        else:
            devset.append(example)
    # shuffle trainset and testset
    random.shuffle(trainset)
    random.shuffle(devset)

    print('GT Dataset Loaded!')
    print(f'Train Set : {len(trainset)}  |  Validation Set : {len(devset)}')

    # Set up the evaluator, which can be re-used in your code.
    evaluator = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)

    print(' Evaluating LLM!')
    # Launch evaluation for the original LLM
    llm_accuracy = evaluator(llm_agent, metric=utils.validate_answer)

    return llm_accuracy


if __name__=='__main__':
    '''
    Checklist for this experiment:
        - config.py 
            - Board Representation: utils.TaskContext
            - Cross Representation: utils.TaskContext
            - Generate Action : utils.GenerateAction
        
    '''

    # configuration
    # llm_model_id = 'ollama_chat/llama2:13b'
    llm_model_id = 'ollama_chat/llama3.2:3b'
    # llm_model_id = 'llama3-70b-8192'
    # GT_dataset = "GT_database/GT_dqn2_1D_int_top1_AA.txt"
    GT_dataset = "GT_database/GT_human.txt"
    result_path = f'experiments_results/LLM_phase/GT_comparison/llama3-70b__1D_int__metric-v2__human_AA.txt'
    result_path = os.path.join(result_path)
    
    llm_accuracy = GT_comparison(llm_model_id, GT_dataset, GROQ=False)
    
    with open(result_path, 'a') as f:
        f.write('Experiment Details:\n')
        f.write('-------------------\n')
        f.write(f'llm_model_id : {llm_model_id}\n')
        f.write(f'GT_dataset : {GT_dataset}\n')
        f.write(f'LLM Accuracy: {llm_accuracy}\n')
        f.write('-------------------\n\n')
        