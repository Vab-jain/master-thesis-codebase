# config.py
'''
Global configurations for all experiments
'''
import os

global board_representation, cross_representation, dspy_signature, validation_metric, task_decription, prompting_method

# accessed in: prompt_tune.py, wrappers.py
board_representation = '1D'      
# board_representation = '2D'

# accessed in: prompt_tune.py, wrappers.py
cross_representation = '1/-1'
# cross_representation = 'X/O'

# defined in llm_config.py
# accessed in: prompt_tune.py, wrappers.py
dspy_signature = 'v2'

# defined in dspy_metric.py
# accessed in: prompt_tune.py
validation_metric = 'v2'

# accessed in wrapper.py
prompting_method = 'CoT' # [ZS, CoT]

# context sample
# ### for 1d board representation (1,-1,0)
# td1 = "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. The board is represented as an array of lenght 9, starting from 0 to 8. Each element represents a cell of the tictactoe board. '0' indicates an empty cell, '1' indicates a cross, and '-1' indicates a circle."
# ### for 1d board representation (X,O,_)
# td2 = "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. The board is represented as an array of lenght 9, starting from 0 to 8. Each element represents a cell of the tictactoe board."
# ### for 2d board representation (1,-1,0)
# td3 = "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. The board is represented as a (3x3) array, with each cell being represented as an integer starting from 0 to 8. '0' indicates an empty cell, '1' indicates a cross, and '-1' indicates a circle."
# ### for 2d board representation (X,O,_)
# td4 = "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. The board is represented as an array of lenght 9, starting from 0 to 8. Each element represents a cell of the tictactoe board."

task_prompts = {
    'td_og' : "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. '0' indicates an empty cell, '1' indicates a cross, and '-1' indicates a circle.",

    # 1D Board, Integer Representation (Full)
    "td1": "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. The board is represented as an array of length 9, starting from 0 to 8. Each element represents a cell of the tic-tac-toe board. '0' indicates an empty cell, '1' indicates a cross, and '-1' indicates a circle.",
    
    # 1D Board, Character Representation (Full)
    "td2": "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. The board is represented as an array of length 9, starting from 0 to 8. Each element represents a cell of the tic-tac-toe board. '_' indicates an empty cell, 'X' indicates a cross, and 'O' indicates a circle.",
    
    # 2D Board, Integer Representation
    "td3": "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. The board is represented as a (3x3) array. '0' indicates an empty cell, '1' indicates a cross, and '-1' indicates a circle.",
    
    # 2D Board, Character Representation
    "td4": "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. The board is represented as a (3x3) array. '_' indicates an empty cell, 'X' indicates a cross, and 'O' indicates a circle.",
    
    # 1D Board, Integer Representation (Missing Board Info)
    "td5": "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. '0' indicates an empty cell, '1' indicates a cross, and '-1' indicates a circle.",
    
    # 1D Board, Integer Representation (No Task Mention)
    "td6": "You are playing a game of tic-tac-toe as a Cross (X). The board is represented as an array of length 9, starting from 0 to 8. Each element represents a cell of the tic-tac-toe board. '0' indicates an empty cell, '1' indicates a cross, and '-1' indicates a circle.",
    
    # 2D Board, Integer Representation (No Game Mention)
    "td7": "Your task is to play the next move and specify the grid cell index where you would place your turn. The board is represented as a (3x3) array. '0' indicates an empty cell, '1' indicates a cross, and '-1' indicates a circle.",
    
    # 2D Board, Character Representation (No Task Mention)
    "td8": "You are playing a game of tic-tac-toe as a Cross (X). The board is represented as a (3x3) array with X, O, and _ symbols.",
    
    # 1D Board, Character Representation (Minimal Instruction)
    "td9": "Your task is to pick the next move in tic-tac-toe. The board is a 1D array with X, O, and _.",
    
    # 1D Board, Integer Representation (Minimalist)
    "td10": "Pick a move. The board is a 1D array of 0, 1, and -1.",
    
    # 2D Board, Integer Representation (Minimalist)
    "td11": "Pick a move. The board is a (3x3) grid of 0, 1, and -1."
}


# defined in wrapper.py
task_decription = task_prompts['td_og']
          

def update_config(board_representation=None, cross_representation=None, task_id=None, prompting_method=None):
    '''
    Update the global configuration variables
    '''
    if board_representation is not None:
        globals()['board_representation'] = board_representation
    if cross_representation is not None:
        globals()['cross_representation'] = cross_representation
    if task_id is not None:
        globals()['task_decription'] = task_prompts[task_id]
    if prompting_method is not None:
        globals()['prompting_method'] = prompting_method


# Get Groq API key from environment variable
# Set it with: export GROQ_API_KEY='your-api-key-here'
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')