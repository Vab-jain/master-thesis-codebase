# utils.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_training_results(win_history, loss_history, draw_history, agent_plays, show_plot=True, save_path=None):
    episodes = range(1, len(win_history) + 1)
    plt.figure(figsize=(12, 10))  # Adjusted figure size for two subplots

    # First subplot: Cumulative results
    plt.subplot(2, 1, 1)
    cumulative_wins = np.cumsum(win_history)
    cumulative_losses = np.cumsum(loss_history)
    cumulative_draws = np.cumsum(draw_history)
    plt.plot(episodes, cumulative_wins, label='Wins')
    plt.plot(episodes, cumulative_losses, label='Losses')
    plt.plot(episodes, cumulative_draws, label='Draws')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Number of Games')

    # Adjust the title based on opponent_first
    if agent_plays=='first_player':
        title = 'Cumulative Training Results (Agent plays first)'
    elif agent_plays == 'second_player':
        title = 'Cumulative Training Results (Agent plays second)'
    else:
        title = 'Cumulative Training Results (Agent plays first or second randomly)'
    plt.title(title)
    plt.legend()

    # Second subplot: Win/Loss/Draw percentages every 100 episodes
    plt.subplot(2, 1, 2)
    window_size = 100
    num_windows = len(win_history) // window_size
    win_percentages = []
    loss_percentages = []
    draw_percentages = []
    window_centers = []

    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window_wins = sum(win_history[start:end])
        window_losses = sum(loss_history[start:end])
        window_draws = sum(draw_history[start:end])
        total_games = window_wins + window_losses + window_draws
        win_percentages.append((window_wins / total_games) * 100)
        loss_percentages.append((window_losses / total_games) * 100)
        draw_percentages.append((window_draws / total_games) * 100)
        window_centers.append(start + window_size // 2 + 1)  # Center of the window

    plt.plot(window_centers, win_percentages, label='Win %', marker='o')
    plt.plot(window_centers, loss_percentages, label='Loss %', marker='o')
    plt.plot(window_centers, draw_percentages, label='Draw %', marker='o')
    plt.xlabel('Episode Number')
    plt.ylabel('Percentage')
    plt.title('Win/Loss/Draw Percentage Every 100 Episodes')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    elif show_plot:
        plt.show()
    else:
        plt.close()

def set_random_seeds(seed=None):
    import random
    import numpy as np
    import torch
    if seed is not None:
        print(f"Setting random seed to: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def state_to_board(state):
    # Convert state vector back to board representation
    state_reshaped = state.reshape(3, 3, 3)
    board = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            if state_reshaped[i, j, 0] == 1:
                board[i, j] = 1
            elif state_reshaped[i, j, 1] == 1:
                board[i, j] = 0
            elif state_reshaped[i, j, 2] == 1:
                board[i, j] = -1
    return board


# Function to read a txt file into a pandas DataFrame
def read_txt_to_dataframe(filename, available_actions=True):
    # Read the tab-separated file into a DataFrame
    # Read the file as a DataFrame
    if available_actions:
        df = pd.read_csv(filename, sep="\t", header=None, names=["board_str", "available_actions", "actions"], dtype=str)
    else:
        df = pd.read_csv(filename, sep="\t", header=None, names=["board_str", "actions"], dtype=str)

    # Drop any NaN rows (in case of missing values)
    df.dropna(subset=["actions"], inplace=True)

    # Ensure 'actions' column is a string and process it
    df["actions"] = df["actions"].astype(str).str.strip("[]").str.split()

    # Convert actions to a list of integers/float
    df["actions"] = df["actions"].apply(lambda x: [int(i) for i in x])

    # take the first element of the list
    df["actions"] = df["actions"].apply(lambda x: x[0])    

    return df