import gymnasium as gym
import random
import numpy as np
import pandas as pd
from scipy import interpolate

# Read the CSV file. Option 1 is red, Option 2 is blue
df_red = pd.read_csv('../path_trace_foraging_red.csv')
df_blue = pd.read_csv('../path_trace_foraging_blue.csv')

# Create probability functions with interpolation for both options
p_red = interpolate.interp1d(df_red['trial'], df_red['p'], kind='linear', fill_value='extrapolate')
p_blue = interpolate.interp1d(df_blue['trial'], df_blue['p'], kind='linear', fill_value='extrapolate')




# env = gym.make('Taxi-v3', render_mode='human')
# env = gym.make('Taxi-v3')

# epsilon = 1.0
# epsilon_decay = 0.9995
# min_epsilon = 0.01
alpha = 0.95
gamma = 0.9
# num_episodes = 10000
max_steps = 300
k = 2

q_table = np.zeros((k,max_steps)) #normally it would be an array of shape (n_states, k), and we would just update the value in place,
# without storing the previous values. But here we don't have states, and we do care about 
# storing the values for each time step so we can plot it later, so we do it like this

#  Since on the paper it says that participants got to know at the beginning if the decks where good, bad, or mediocre, the RL algorithm
# can have also that information through the bias (the first q_value for each deck).But interestingly they do not do that on the paper,
# they give both options an initial value of 0.5
# q_table[0,0] = 1
# q_table[1,0] = 0
q_table[:,0] = 0.5

beta=50 #medium noise

# v = np.zeros((n_decks,max_pulls))
def choose_option(time):
    p_choose_red = 1/(1 + np.exp(-beta*(q_table[0,time]-q_table[1,time])))
    if random.uniform(0,1) < p_choose_red:
        return 0, p_red(time)/100.0, 1
    else:
        return 1, p_blue(time)/100.0, 0
    
def get_reward(probability):
    if random.uniform(0,1) < probability:
        return 1
    else:
        return 0

print('Begin training!')
for step in range(max_steps-1):
    option, option_prob_value, unchosen_option = choose_option(step)
    reward = get_reward(option_prob_value)
    q_table[option,step+1] = q_table[option,step]+alpha*(reward - np.max(q_table[option,step]))
    q_table[unchosen_option,step+1] = q_table[unchosen_option,step]

print('Finished training!')




