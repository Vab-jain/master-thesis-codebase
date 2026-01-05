'''
##### This file contains the validation meteric for prompt finetuning (all versions). 
##### Select/uncomment the appropiate version for different experiments in 'run_LLM_experiment.py' or 'prompt_tune.py'
'''
import config
import numpy as np

if config.validation_metric=='v3':
    '''
    Validation Metric: v3
        Normalizes the labels->qvalues to [0,1] generating a score_list
        Returns the score from score_list corresponding to the selected action 
    '''

    def validate_answer(label, pred, trace=None):
        def normalize_list(values):
            # Convert list to numpy array
            values = np.array(values, dtype=float)

            # Check if all values are -inf
            if np.all(np.isneginf(values)):
                return np.zeros_like(values).tolist()  # Return all zeros

            # Replace -inf with the minimum finite value
            finite_values = values[np.isfinite(values)]
            if len(finite_values) == 0:
                return np.zeros_like(values).tolist()  # If all values are -inf, return zeros

            min_val = np.min(finite_values)
            values[np.isneginf(values)] = min_val  # Replace -inf with the smallest finite value

            # Min-Max normalization
            min_val = np.min(values)
            max_val = np.max(values)

            if min_val == max_val:
                return np.zeros_like(values).tolist()  # Avoid division by zero

            normalized = (values - min_val) / (max_val - min_val)
            return normalized.tolist()

        # Convert label.answer to a list of floats (always length 9)
        try:
            q_values = list(map(float, label.answer.split()))
            if len(q_values) != 9:
                raise ValueError("label.answer must contain exactly 9 values")
        except ValueError:
            raise ValueError(f"Invalid label.answer format: {label.answer}")

        # Normalize q_values
        score_list = normalize_list(q_values)
        
        # return bool exact match if we are bootstraping
        if trace:
            return int(max(q_values)) == int(pred.answer)

        # Convert pred.answer to an integer
        try:
            ans = int(pred.answer)
        except ValueError:
            return 0  # If pred.answer is not a valid integer, return 0

        # Ensure ans is within the valid range (0-8)
        if ans < 0 or ans > 8:
            return 0

        # Return the corresponding normalized score
        return score_list[ans]

    

if config.validation_metric=='v2':
    '''
    Validation Metric: v2
        Compares if the LLM predicted action is same as the ground truth action.
        GT-Database: GT_dqn_single_action__1D_int / top3_action__1D_int
    '''
    def validate_answer(label, pred, trace=None):
        def get_symmetric_positions(board, index):
            """
            Given the current state of the Tic-Tac-Toe board and a cell index,
            returns a list of cell positions that are equivalent to play in.
            """
            symmetry_pairs = {
                "horizontal": [[0, 6], [1, 7], [2, 8]],
                "vertical": [[0, 2], [3, 5], [6, 8]],
                "diagonal_1": [[0, 8], [1, 5], [3, 7]],
                "diagonal_2": [[2, 6], [1, 3], [5, 7]],
            }
            
            symmetry_checks = {
                "horizontal": board[0] == board[6] and board[1] == board[7] and board[2] == board[8],
                "vertical": board[0] == board[2] and board[3] == board[5] and board[6] == board[8],
                "diagonal_1": board[0] == board[8] and board[1] == board[5] and board[3] == board[7],
                "diagonal_2": board[2] == board[6] and board[1] == board[3] and board[5] == board[7],
            }
            
            restricted_symmetries = {
                4: [],
                1: ["horizontal", "diagonal_1", "diagonal_2"],
                7: ["horizontal", "diagonal_1", "diagonal_2"],
                3: ["vertical", "diagonal_1", "diagonal_2"],
                5: ["vertical", "diagonal_1", "diagonal_2"],
                2: ["horizontal", "vertical", "diagonal_2"],
                6: ["horizontal", "vertical", "diagonal_2"],
                0: ["horizontal", "vertical", "diagonal_1"],
                8: ["horizontal", "vertical", "diagonal_1"],
            }

            symmetric_positions = {index}
            for symmetry in restricted_symmetries[index]:
                if symmetry_checks[symmetry]:
                    symmetric_positions.update(i for pair in symmetry_pairs[symmetry] if index in pair for i in pair)

            return list(symmetric_positions)

        # Convert current_state to a list of integers
        board = list(map(int, label.current_state.split()))

        # Initialize an empty set to store all symmetric positions
        all_symmetric_positions = set()

        # Check if label.answer is a list or a single integer
        if isinstance(label.answer, list):
            for ans in label.answer:
                all_symmetric_positions.update(get_symmetric_positions(board, int(ans)))
        else:
            all_symmetric_positions.update(get_symmetric_positions(board, int(label.answer)))

        # Check if the predicted cell is in the symmetric positions of the label cell
        if int(pred.answer) in all_symmetric_positions:
            return True
        return False



if config.validation_metric=='v1':
    '''
    Validation Metric: v1
        Compares if the LLM predicted action is same as the ground truth action.
        GT-Database: GT_dqn_single_action_1D
    '''
    def validate_answer(label, pred, trace=None):
        return int(label.answer[0]) == int(pred.answer)