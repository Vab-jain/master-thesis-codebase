import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


import pandas as pd

def calculate_mean_return(csv_files):
    """
    Given multiple CSV file paths, this function reads the 'average_return' column from each file,
    calculates the mean for each row across the files (excluding the last 2 rows), 
    and returns the values as a list.

    :param csv_files: List of file paths to CSVs
    :return: List of mean values for each row across all files (excluding the last 2 rows)
    """
    dataframes = [pd.read_csv(file, usecols=['average_return']) for file in csv_files]  # Exclude last 2 rows
    combined_df = pd.concat(dataframes, axis=1)
    mean_values = combined_df.mean(axis=1).tolist()  # Calculate row-wise mean
    std_values = combined_df.std(axis=1).tolist()  # Calculate row-wise standard deviation
    
    return mean_values, std_values

def plot_logs(file1, file2, label1='Log 1', label2='Log 2', output_file='comparison_plot_return.png'):
    # Load data from CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Extract relevant columns
    checkpoints1 = df1['checkpoint']
    checkpoints2 = df2['checkpoint']
    
    # Calculate average return
    avg_return1 = (df1['win'] - df1['loss']) / 100
    avg_return2 = (df2['win'] - df2['loss']) / 100
    
    print(avg_return1)
    print(avg_return2)
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    
    # Plot average return for both logs
    plt.plot(checkpoints1, avg_return1, label=f'Average Return ({label1})', marker='o', linestyle='-')
    plt.plot(checkpoints2, avg_return2, label=f'Average Return ({label2})', marker='s', linestyle='--')
    
    # Labels and title
    plt.xlabel('Checkpoint')
    plt.ylabel('Average Return')
    plt.title('Average Return Comparison')
    plt.legend()
    plt.grid()
    
    # Save the plot
    plt.savefig(output_file)
    plt.show()

# # Example usage:
# exp1 = './experiments_results/RL_phase/EA-DQN3/sample_efficiency.csv'
# exp2 = './experiments_results/RL_phase/E16-sig_v2__1D__LLM70b_GROQ_seed3/sample_efficiency.csv'
# plot_logs(exp1, exp2, 'DQN', 'DQN-with LLM')


def plot_win_percentage(win_percentage1, win_percentage2, label1='Model A', label2='Model B', output_file='win_percentage_plot.png'):
    checkpoints = [i for i in range(100,1000,100)]
    checkpoints.append('final')
    
    plt.figure(figsize=(10, 5))
    plt.plot(checkpoints, win_percentage1, label=f'Win % ({label1})', marker='o', linestyle='-')
    plt.plot(checkpoints, win_percentage2, label=f'Win % ({label2})', marker='s', linestyle='--')
    
    plt.xlabel('Training Episodes')
    plt.ylabel('Win Percentage')
    plt.title('Win Percentage Comparison')
    plt.legend()
    plt.grid()
    
    # Save the plot
    plt.savefig(output_file)
    plt.show()
 


def plot_means_with_std(data, labels=None, colors=None, x_values=None, x_tick_labels=None, output_file='average_return_plot.png'):
    """
    Plots multiple mean and std pairs with shaded standard deviation areas.
    
    Parameters:
    - data: List of tuples [(mean_list1, std_list1), (mean_list2, std_list2), ...]
    - labels: List of labels for each line (default: None)
    - colors: List of colors for each line (default: None)
    - x_values: Custom x-axis values (default: range of the first mean list)
    - x_tick_labels: List of labels for x-ticks (default: None)
    """
    plt.figure(figsize=(8, 6))
    
    if labels is None:
        labels = [f"Series {i+1}" for i in range(len(data))]
    
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))  # Generate distinct colors

    for i, ((mean, std), label, color) in enumerate(zip(data, labels, colors)):
        mean = np.array(mean)
        std = np.array(std)
        x = np.arange(len(mean)) if x_values is None else np.array(x_values)
        
        plt.plot(x, mean, label=label, color=color)
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.1)  # Softer shading
    
    plt.xlabel("Number of Episodes")
    plt.ylabel("Average Return")
    plt.legend()
    plt.title("Average Return Comparison")
    plt.grid(True)

    if x_tick_labels is not None:
        plt.xticks(ticks=np.arange(len(x_tick_labels)), labels=x_tick_labels, rotation=45)
    
    plt.savefig(output_file)
    plt.show()




path_dqn = ['experiments_results/RL_phase/EA-DQN42/sample_efficiency.csv', 'experiments_results/RL_phase/EA-DQN50/sample_efficiency.csv', 'experiments_results/RL_phase/EA-DQN70/sample_efficiency.csv']
path_llm_int = ['experiments_results/RL_phase/EF-llama3-70b-1D_int-CoT_seed50/sample_efficiency.csv', 'experiments_results/RL_phase/EF-llama3-70b-1D_int-CoT/sample_efficiency.csv', 'experiments_results/RL_phase/EF-llama3-70b-1D_int-CoT_seed70/sample_efficiency.csv']
# path_llm_oh = ['experiments_results/RL_phase/EF-llama3-70b-1D_int-CoT-OH/sample_efficiency.csv', 'experiments_results/RL_phase/EF-llama3-70b-1D_int-CoT-OH_seed50/sample_efficiency.csv', 'experiments_results/RL_phase/EF-llama3-70b-1D_int-CoT-OH_seed70/sample_efficiency.csv']
path_llm_50 = ['experiments_results/RL_phase/EF-llama3-70b-1D_int-llm50-CoT_seed50/sample_efficiency.csv', 'experiments_results/RL_phase/EF-llama3-70b-1D_int-llm50-CoT/sample_efficiency.csv', 'experiments_results/RL_phase/EF-llama3-70b-1D_int-llm50-CoT_seed70/sample_efficiency.csv']

    
mean_dqn, std_dqn = calculate_mean_return(path_dqn)
mean_llm_int, std_llm_int = calculate_mean_return(path_llm_int)
mean_llm_oh, std_llm_oh = calculate_mean_return(path_llm_50)

plot_means_with_std([(mean_dqn, std_dqn), (mean_llm_int, std_llm_int), (mean_llm_oh, std_llm_oh)], 
                    labels=['DQN', 'DQN with LLM (prob:1)', 'DQN with LLM (prob:0.5)'], 
                    colors=['blue', 'red', 'green'],
                    x_tick_labels=['100', '200', '300', '400', '500', '600', '700', '800', '900', '1000'],
                    output_file='comparison_llm_freq.png')
# plot_win_percentage([23,76,74,78], [39,77,79,82], 'DQN', 'DQN-with LLM')
# plot_win_percentage([-0.10,0.59,0.61,0.69], [0.02,0.61,0.69,0.71], 'DQN', 'DQN-with LLM')