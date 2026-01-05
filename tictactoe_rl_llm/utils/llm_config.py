import dspy
import config

class TaskContext():
    def __init__(self, 
                 board_representation=config.board_representation, 
                 cross_representataion=config.cross_representation):
        self.context = ''
        if board_representation=='1D':
            if cross_representataion=='1/-1':
                self.context = "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. '0' indicates an empty cell, '1' indicates a cross, and '-1' indicates a circle.",
            if cross_representataion=='X/O':
                    self.context = "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. '_' indicates an empty cell, 'X' indicates a cross, and 'O' indicates a circle."
        
        # sanity check for self.context
        if self.context=='':
            raise Exception("Something went wrong while generating TaskContext. Please check config.py")
          

if config.dspy_signature=='v1':
    # version: v1
    # for output as single action between [0-8]
    class GenerateAction(dspy.Signature):
        context = dspy.InputField(desc="task description")
        current_state = dspy.InputField(desc="current state represented as an array of lenght 9, starting from 0 to 8.")
        answer: int = dspy.OutputField(desc="an integer describing next action given current state")

elif config.dspy_signature=='v2':
    class GenerateAction(dspy.Signature):
        context = dspy.InputField(desc="task description")
        current_state = dspy.InputField(desc="current state represented as an array of lenght 9, starting from 0 to 8.")
        available_actions = dspy.InputField(desc="list of empty cells avaiable to play")
        answer: int = dspy.OutputField(desc="an integer describing next action given current state")

elif config.dspy_signature=='v3':
    class GenerateAction(dspy.Signature):
        current_state = dspy.InputField(desc="current state represented as an array of lenght 9, starting from 0 to 8.")
        available_actions = dspy.InputField(desc="list of empty cells avaiable to play")
        answer: int = dspy.OutputField(desc="an integer describing next action given current state")