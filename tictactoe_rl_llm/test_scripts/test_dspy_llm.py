import dspy
from utils import state_to_board
from llm_utils import get_board_representation

lm = dspy.LM('ollama_chat/llama3.2:3b', api_base='http://localhost:11434')
dspy.configure(lm=lm)

class GenerateAction(dspy.Signature):
    context = dspy.InputField(desc="task description")
    current_state = dspy.InputField(desc="current state")
    answer: int = dspy.OutputField(desc="an integer describing next action given current state")

llm_agent = dspy.ChainOfThought(GenerateAction)
# llm_agent.load(path='optim_llama3_3b.pkl')

context_sample = "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. The board is represented as an array of lenght 9, starting from 0 to 8. Each element represents a cell of the tictactoe board. '0' indicates an empty cell, '1' indicates a cross, and '-1' indicates a circle."

def get_llm_suggestion(state, save_llm_cache=False):
    board = state_to_board(state)
    board_str, _ = get_board_representation(board) 
    response = llm_agent(context=context_sample, current_state=board_str)
    return int(response.answer)