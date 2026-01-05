class TicTacToeOracle: 
    def __init__(self): 
        self.winning_combinations = [ 
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows 
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns 
            [0, 4, 8], [2, 4, 6]              # Diagonals 
        ]   

    def select_action(self, board_state): 
        """ 
        Suggest the best next move for the player (1 or -1) based on the current board state. 
 
        Args: 
            board_state (list): A list of 9 elements representing the current Tic-Tac-Toe board. 
                                Each element can be 1, -1, or 0 (empty space). 
            player_turn (int): The current player's turn (1 for X, -1 for O). 
  
        Returns: 
            int: The index of the grid cell (0-8) where the player should place their mark. 
        """ 
        board_state = board_state.flatten()
        player = -1
        opponent = 1 
        
        # Step 1: Check if we can win in the next move 
        for index in self.get_available_moves(board_state): 
            if self.is_winning_move(board_state, index, player): 
                return index 

        # Step 2: Check if the opponent can win in the next move and block them 
        for index in self.get_available_moves(board_state): 
            if self.is_winning_move(board_state, index, opponent): 
                return index 
        

        # Step 3: Take the center if available 
        if board_state[4] == 0: 
            return 4 
         
        # Step 4: Take a corner if available 
        for index in [0, 2, 6, 8]: 
            if board_state[index] == 0: 
                return index 
         
        # Step 5: Take any remaining available space 
        available_moves = self.get_available_moves(board_state) 
        if available_moves: 
            return available_moves[0] 
        else: 
            return -1

     
    def get_available_moves(self, board_state): 
        """Return a list of available moves (empty spaces).""" 
        return [i for i, cell in enumerate(board_state) if cell == 0] 
  
    def is_winning_move(self, board_state, index, player): 
        """Check if placing 'player' at 'index' would result in a win.""" 
        temp_board = board_state.copy() 
        temp_board[index] = player 
        return any(all(temp_board[i] == player for i in combo) for combo in self.winning_combinations) 
  

if __name__ == "__main__":  
    # Example usage 
    from env import TicTacToeEnv
    import numpy as np
    
    
    opponent_first = True

    env = TicTacToeEnv(opponent_first=opponent_first)

    oracle = TicTacToeOracle() 
    
    for episode in range(10):
        print("\n\n~~~~ New Episode begins ~~~~")
        state = env.reset()
        # env.render()
        done = False
        while not done:
            env.render()
            action = oracle.select_action(board_state=state, player_turn=1)
            
            print(f"Oracle suggested: {action}")
            state, reward, done, _ = env.step(action)
            
            input()
            
    
    # board_state = ['X', 'O', 'X', '', '', 'O', '', 'X', 'O'] 
    # player_turn = 'X' 
    
    # suggested_move = oracle.get_oracle_suggestion(board_state, player_turn) 
    # print(f"Oracle suggests placing at index: {suggested_move}") 
 