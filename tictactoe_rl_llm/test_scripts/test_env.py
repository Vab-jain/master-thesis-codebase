from env import TicTacToeEnv
import numpy as np

env = TicTacToeEnv()

for episode in range(10):
    first_player = np.random.choice([1,-1])
    state = env.reset(first_player)
    # env.render()
    done = False
    print(f'Begin Episode: {episode+1}')
    while not done:
        env.render()
        action = int(input('Enter action: '))
        state, reward, done, _ = env.step(action)
    if reward==1:
        print('You won!')
    elif reward==-1:
        print('You lost!')
    elif reward==-10:
        print('Invalid Action!')
    else:
        print('Draw!')
    env.render()