import gymnasium as gym
import random
import numpy as np

# env = gym.make('Taxi-v3', render_mode='human')
env = gym.make('Taxi-v3')

epsilon = 1.0
epsilon_decay = 0.9995
min_epsilon = 0.01
alpha = 0.95
gamma = 0.9
num_episodes = 10000
max_steps = 100

q_table = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state, epsilon):
    if random.uniform(0,1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state, :])
    return action

print('Begin training!')
for episode in range(num_episodes):
    state, _ = env.reset()

    done = False

    for step in range(max_steps):
        action = choose_action(state, epsilon)
        new_state, reward, done, truncated, info = env.step(action) 
        q_table[state,action] = (1-alpha)*q_table[state,action]+alpha*(reward + gamma*np.max(q_table[new_state,:]))
        
        state = new_state
        if done or truncated:
            break

    epsilon = max(min_epsilon, epsilon*epsilon_decay)
print('Finished training!')

env = gym.make('Taxi-v3', render_mode='human')
for episode in range(1):
    state, _ = env.reset()
    done = False
    print('Episode', episode)
    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state, :])
        new_state, reward, done, truncated, info = env.step(action)
        print(reward)
        state = new_state
        if done or truncated:
            env.render()
            print(f'Finished episode {episode}, with reward {reward}')
            break



