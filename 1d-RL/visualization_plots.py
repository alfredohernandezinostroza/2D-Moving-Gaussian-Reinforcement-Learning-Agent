
import mlflow
import mlflow.tensorflow
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
import os
import time
from pathlib import Path
from urllib.parse import quote
from functools import wraps
from RLGaussianLibrary import REINFORCEAgent, Environment

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        try:
            start_time = time.process_time()
            result = func(*args, **kwargs)
            end_time = time.process_time()
            total_time = end_time - start_time
            print(f'Function {func.__name__} Took {total_time:.4f} seconds')
            return result
        except KeyboardInterrupt:
            print(f'\nFunction {func.__name__} interrupted.')
            raise
    return timeit_wrapper

@timeit
def train_reinforce_plot(n_episodes=5000, max_steps=20, gamma=0.99, visualize_every=50):
    """Train REINFORCE agent"""
    
    # Initialize environment and agent
    env = Environment()
    agent = REINFORCEAgent()
    
    # Set up visualization
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flat
    axes = [ax1, ax1.twinx(), ax2, ax3, ax4]
    fig.suptitle('REINFORCE Training Progress')
    
    for episode in range(n_episodes):
        # Storage for this episode
        states = []
        actions = []
        rewards = []
        
        # Reset environment
        state = env.reset()
        
        # Generate episode
        for step in range(max_steps):
            states.append(state)
            
            # Sample action from policy
            action = agent.sample_action(state)
            actions.append(action)
            
            # Get reward and transition
            reward = env.reward(state)
            rewards.append(reward)
            
            # Move to next state
            state = env.transition(state, action)
        
        # Compute returns and apply baseline
        returns = compute_returns(rewards, gamma)
        returns = apply_baseline(returns)
        
        # Update policy
        agent.update_policy(states, actions, returns)
        
        # Store results
        agent.episode_rewards.append(sum(rewards))
        agent.policy_params.append([agent.w1, agent.b1, agent.log_std])
        
        # Visualization
        if episode % visualize_every == 0 or episode == 0:
            axes[1].clear()
            visualize_training(env, agent, episode, states, fig, axes)
            plt.pause(0.001)
    
    plt.ioff()
    return env, agent
def visualize_training(env, agent, episode, last_states, fig, axes):
    """Visualize training progress"""
    
    # Clear all subplots and their twin axes
    # for ax in axes.flat:
    for ax in axes:
        ax.clear()
        # Also clear any twin axes that might exist
        # if hasattr(ax, 'twin_axes'):
        #     for twin_ax in ax.twin_axes:
        #         twin_ax.clear()
    ax1, ax1_twin, ax2, ax3, ax4 = axes
    
    # Plot 1: Environment and Policy
    # ax1 = axes[0, 0]
    x_range = np.linspace(env.state_min, env.state_max, 100)
    r_range = [env.reward(x) for x in x_range]
    ax1.plot(x_range, r_range, 'b-', linewidth=2, label='Reward Function')
    
    # Plot policy mean - create fresh twin axis each time
    # ax1_twin = ax1.twinx()
    mu_range = [agent.policy_mean(x) for x in x_range]
    ax1_twin.plot(x_range, mu_range, 'r-', linewidth=2, label='Policy Mean')
    ax1_twin.set_ylabel('Policy Mean Action', color='r')
    ax1_twin.set_ylim([agent.action_min, agent.action_max])
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    ax1.set_ylabel('Reward', color='b')
    ax1.set_xlabel('State')
    ax1.set_title(f'Environment and Learned Policy (Episode {episode})')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Plot 2: Episode trajectory
    # ax2 = axes[0, 1]
    if last_states:
        ax2.plot(last_states, 'o-', linewidth=1.5, label='States')
        ax2.axhline(y=env.reward_center, color='r', linestyle='--', label='Optimal State')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('State')
    ax2.set_title('Last Episode Trajectory')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Learning curve
    # ax3 = axes[1, 0]
    if agent.episode_rewards:
        ax3.plot(agent.episode_rewards, linewidth=1.5)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Total Reward')
    ax3.set_title('Learning Progress')
    ax3.grid(True)
    
    # Plot 4: Parameter evolution
    # ax4 = axes[1, 1]
    if agent.policy_params:
        params = np.array(agent.policy_params)
        ax4.plot(params[:, 0], 'r-', linewidth=1.5, label='w1')
        ax4.plot(params[:, 1], 'g-', linewidth=1.5, label='b1')
        ax4.plot(np.exp(params[:, 2]), 'b-', linewidth=1.5, label='std')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Parameter Value')
    ax4.set_title('Policy Parameters')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
@timeit
def test_learned_policy_plot(env, agent, show=True):
    """Test and visualize the learned policy"""
    
    print("\n=== REINFORCE Training Complete ===")
    print("Final parameters:")
    print(f"  w1 = {agent.w1:.3f}")
    print(f"  b1 = {agent.b1:.3f}")
    print(f"  std = {agent.policy_std():.3f}")
    
    if len(agent.episode_rewards) >= 50:
        avg_reward = np.mean(agent.episode_rewards[-50:])
        print(f"\nFinal average reward: {avg_reward:.3f}")
    
    # Create test visualization
    fig = plt.figure(figsize=(12, 8))

    # Define a 2x2 grid
    gs = fig.add_gridspec(2, 2)

    # Create subplots
    ax1 = fig.add_subplot(gs[0, :])     # spans row 0, all columns → wide plot
    ax2 = fig.add_subplot(gs[1, 0])     # row 1, column 0
    ax3 = fig.add_subplot(gs[1, 1])   

    fig.suptitle('Final Policy Evaluation')
    
    # Test trajectories
    test_states = np.linspace(env.state_min, env.state_max, 5)
    colors = plt.cm.tab10(np.linspace(0, 1, len(test_states)))
    
    # Plot 1: Trajectories on reward landscape
    # ax1 = axes[0, :]
    x_range = np.linspace(env.state_min, env.state_max, 100)
    r_range = [env.reward(x) for x in x_range]
    ax1.plot(x_range, r_range, 'b-', linewidth=3, label='Reward Function')
    
    for i, initial_state in enumerate(test_states):
        state = initial_state
        trajectory = [state]
        
        for step in range(10):
            action = agent.policy_mean(state)  # Use mean action for testing
            state = env.transition(state, action)
            trajectory.append(state)
        
        # Plot trajectory points on reward landscape
        trajectory_rewards = [env.reward(s) for s in trajectory]
        ax1.plot(trajectory, trajectory_rewards, 'o-', 
                color=colors[i], linewidth=2, markersize=6)
        
        # Add direction arrow
        if len(trajectory) > 1:
            ax1.annotate('', xy=(trajectory[1], env.reward(trajectory[1])), 
                        xytext=(trajectory[0], env.reward(trajectory[0])),
                        arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
    
    ax1.axvline(x=env.reward_center, color='r', linestyle='--', linewidth=2, label='Optimal State')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Reward')
    ax1.set_title('Agent Trajectories on Reward Landscape')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: State evolution over time
    # ax2 = axes[1, 0]
    for i, initial_state in enumerate(test_states):
        state = initial_state
        trajectory = [state]
        
        for step in range(10):
            action = agent.policy_mean(state)
            state = env.transition(state, action)
            trajectory.append(state)
        
        ax2.plot(range(len(trajectory)), trajectory, 'o-', 
                color=colors[i], linewidth=2, label=f'Start: {initial_state:.1f}')
    
    ax2.axhline(y=env.reward_center, color='r', linestyle='--', linewidth=2, label='Optimal')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('State')
    ax2.set_title('State Evolution Over Time')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Policy visualization
    # ax3 = axes[1, 1]
    states_plot = np.linspace(env.state_min, env.state_max, 100)
    mu_plot = [agent.policy_mean(s) for s in states_plot]
    sigma_plot = agent.policy_std()
    
    ax3.plot(states_plot, mu_plot, 'r-', linewidth=2, label='Mean')
    ax3.fill_between(states_plot, 
                     np.array(mu_plot) - sigma_plot, 
                     np.array(mu_plot) + sigma_plot, 
                     alpha=0.3, color='r', label='±1 std')
    ax3.set_xlabel('State')
    ax3.set_ylabel('Action')
    ax3.set_title('Learned Gaussian Policy')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig
