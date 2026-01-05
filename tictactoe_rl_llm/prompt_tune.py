import dspy
from dspy.evaluate import Evaluate
import random
import sys
import io
import os

import config
import utils
from utils import GenerateAction, TaskContext



def prompt_tune_llm(
                llm_model_id,
                GT_dataset,
                llm_save_path,
                board_representation=config.board_representation,
                cross_representataion=config.cross_representation,
                GROQ=False
                ):
    '''
    Function to optimize LLM
    
    Args:
        llm_model_id  : 
        GT_dataset  :
        llm_save_path :
        board_representation : (default) '1D',
        cross_representataion : (default) '1/-1'
    
    Return:
        llm_accuracy : 
        optim_llm_accuracy : 
        (saves optimied LLM at 'llm_save_path')
    '''
    
    # configure LLM
    if GROQ:
        lm = dspy.LM(llm_model_id, api_base='https://api.groq.com/openai/v1', api_key=config.GROQ_API_KEY)
    else:
        lm = dspy.LM(llm_model_id, api_base='http://localhost:11434')
    dspy.configure(lm=lm)
    llm_agent = dspy.ChainOfThought(GenerateAction)

    print('LLM Configured!')

    # Create the Task-Context
    taskcontext = TaskContext(board_representation,cross_representataion,)

    # load the dataset
    df = utils.read_txt_to_dataframe(GT_dataset)
    trainset, devset = [], []
    train_size = int(0.2 * len(df))  # First 20% for trainset
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

    print(' Evaluating Original LLM (before optimization)!')
    # Launch evaluation for the original LLM
    llm_accuracy = evaluator(llm_agent, metric=utils.validate_answer)

    # LLM Optimization / Prompt Tuning
    # tp = dspy.MIPROv2(metric=utils.validate_answer, auto="light", num_threads=1, )
    # optimized_llm_agent = tp.compile(llm_agent.deepcopy(), 
    #                                  trainset=trainset, 
    #                                  max_bootstrapped_demos=1,
    #                                  max_labeled_demos=1,
    #                                  requires_permission_to_run=False)
    optimizer = dspy.BootstrapFewShot(metric=utils.validate_answer, 
                                      max_bootstrapped_demos=5,
                                        max_labeled_demos=5,
                                        max_rounds=10,
                                      )
    optimized_llm_agent = optimizer.compile(llm_agent.deepcopy(), trainset=trainset)
    
    print(' Evaluating Optimized LLM (after optimization)!')
    # Launch evaluation for the optimized LLM
    optim_llm_accuracy = evaluator(optimized_llm_agent, metric=utils.validate_answer)

    # save optim_LLM
    optimized_llm_agent.save(llm_save_path)
    print('LLM saved successfully at: ', llm_save_path)

    return llm_accuracy, optim_llm_accuracy


if __name__=='__main__':
    '''
    Checklist for this experiment:
        - config.py 
            - Board Representation: utils.TaskContext
            - Cross Representation: utils.TaskContext
            - Generate Action : utils.GenerateAction
        
    '''

    # configuration
    llm_model_id = 'llama3-70b-8192'
    GT_dataset = "GT_database/GT_dqn2_1D_int_top1_AA.txt"
    root_path = 'saved_optim_llm/optim_llama3-70b__dqn2__1D_int__metricV2__top1__AA_FS5'
    os.makedirs(root_path, exist_ok=True)
    llm_save_path = os.path.join(f'{root_path}/optim_llm.pkl')
    result_path = os.path.join(f'{root_path}/logs.txt')
    
    
    
    llm_accuracy, optim_llm_accuracy = prompt_tune_llm(llm_model_id, GT_dataset, llm_save_path, GROQ=True)

    # Create a StringIO object
    output_buffer = io.StringIO()
    # Redirect stdout to the buffer
    sys.stdout = output_buffer
    # Print something (this will be captured)
    dspy.inspect_history(n=1)
    # Reset stdout to default
    sys.stdout = sys.__stdout__
    # Get the recorded output as a string
    prompt_history = output_buffer.getvalue()
    
    
    with open(result_path, 'w') as f:
        f.write('Experiment Details:\n')
        f.write('-------------------\n')
        f.write(f'llm_model_id : {llm_model_id}\n')
        f.write(f'GT_dataset : {GT_dataset}\n')
        f.write(f'llm_save_path : {llm_save_path}\n\n')
        f.write(f'LLM Accuracy (before optimization): {llm_accuracy}\n')
        f.write(f'Optimized LLM Accuracy (after optimization): {optim_llm_accuracy}\n\n')
        f.write('-------------------\n\n')
        f.write('Prompt Tuning History:\n')
        f.write('-------------------\n')
        f.write(prompt_history)
        f.write('-------------------\n\n')
        