"""
REINFORCE Algorithm Demo - 1D Continuous Control (Python Version)
This demo shows how policy gradient methods work using REINFORCE
on a simple 1D problem where the agent must find the peak of a Gaussian reward
FIXED: Proper handling of large action spaces [-10, 10] and MLflow logging
"""
import mlflow
import mlflow.tensorflow
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
import os
import time
from functools import wraps

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

# Set random seed for reproducibility
np.random.seed(42)

class REINFORCEAgent:
    def __init__(self, action_min=-10, action_max=10, lr_mean=0.005, lr_std=0.0005):
        """Initialize REINFORCE agent with proper scaling for large action spaces"""
        self.action_min = action_min
        self.action_max = action_max
        
        # Policy parameters - FIXED: Smaller initial weights for large action space
        self.w1 = 0.01  # Reduced from 0.1
        self.b1 = 0.0   # Bias
        self.log_std = math.log(2.0)  # Increased initial exploration
        
        # Learning rates - ADJUSTED for larger action space
        self.lr_mean = lr_mean
        self.lr_std = lr_std
        
        # Storage for training history
        self.episode_rewards = []
        self.policy_params = []  # [w1, b1, log_std] for each episode
        
    def policy_mean(self, state):
        """Compute policy mean action for given state"""
        return self.action_max * math.tanh(self.w1 * state + self.b1)
    
    def policy_std(self):
        """Get policy standard deviation"""
        return math.exp(self.log_std)
    
    def sample_action(self, state):
        """Sample action from Gaussian policy"""
        mu = self.policy_mean(state)
        sigma = self.policy_std()
        action = mu + sigma * np.random.randn()
        # Clip to valid action range
        return np.clip(action, self.action_min, self.action_max)
    
    def log_prob(self, state, action):
        """Compute log probability of action given state"""
        mu = self.policy_mean(state)
        sigma = self.policy_std()
        return -0.5 * math.log(2 * math.pi) - self.log_std - 0.5 * ((action - mu) / sigma)**2
    
    def update_policy(self, states, actions, returns):
        """Update policy parameters using REINFORCE"""
        if len(states) == 0:
            return
        
        # Compute gradients
        grad_w1 = 0
        grad_b1 = 0
        grad_log_std = 0
        
        for t, (state_t, action_t, return_t) in enumerate(zip(states, actions, returns)):
            # Recompute policy for this state-action pair
            mu_t = self.policy_mean(state_t)
            sigma_t = self.policy_std()
            
            # Gradient of log probability w.r.t. mean
            d_log_prob_d_mu = (action_t - mu_t) / (sigma_t**2)
            
            # Gradient of mean w.r.t. parameters (chain rule)
            tanh_term = math.tanh(self.w1 * state_t + self.b1)
            d_tanh = 1 - tanh_term**2  # Derivative of tanh
            d_mu_d_w1 = self.action_max * d_tanh * state_t
            d_mu_d_b1 = self.action_max * d_tanh
            
            # Gradient of log probability w.r.t. log_std
            d_log_prob_d_log_std = ((action_t - mu_t)**2 / sigma_t**2 - 1)
            
            # REINFORCE gradients: grad = return * grad_log_prob
            grad_w1 += return_t * d_log_prob_d_mu * d_mu_d_w1
            grad_b1 += return_t * d_log_prob_d_mu * d_mu_d_b1
            grad_log_std += return_t * d_log_prob_d_log_std
        
        # Average gradients and apply gradient clipping
        grad_clip = 1.0
        grad_w1 = np.clip(grad_w1 / len(states), -grad_clip, grad_clip)
        grad_b1 = np.clip(grad_b1 / len(states), -grad_clip, grad_clip)
        grad_log_std = np.clip(grad_log_std / len(states), -grad_clip, grad_clip)
        
        # Update parameters (gradient ascent)
        self.w1 += self.lr_mean * grad_w1
        self.b1 += self.lr_mean * grad_b1
        self.log_std += self.lr_std * grad_log_std
        
        # Prevent log_std from becoming too extreme
        self.log_std = np.clip(self.log_std, math.log(0.1), math.log(5.0))

class Environment:
    def __init__(self, state_min=-5, state_max=5, reward_center=2.0, reward_std=1.0):
        """Initialize 1D continuous environment"""
        self.state_min = state_min
        self.state_max = state_max
        self.reward_center = reward_center
        self.reward_std = reward_std
    
    def reward(self, state):
        """Gaussian reward function"""
        return math.exp(-0.5 * ((state - self.reward_center) / self.reward_std)**2)
    
    def transition(self, state, action):
        """State transition function"""
        new_state = state + action
        return np.clip(new_state, self.state_min, self.state_max)
    
    def reset(self):
        """Reset environment to random initial state"""
        return np.random.uniform(self.state_min, self.state_max)

def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns"""
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return np.array(returns)

def apply_baseline(returns):
    """Apply baseline subtraction and normalization"""
    if len(returns) <= 1:
        return np.zeros_like(returns)
    
    # Baseline subtraction
    baseline = np.mean(returns)
    returns = returns - baseline
    
    # Optional normalization
    if np.std(returns) > 1e-8:
        returns = returns / np.std(returns)
    
    return returns

@timeit
def train_reinforce(n_episodes=5000, max_steps=20, gamma=0.99, visualize_every=1):
    """Train REINFORCE agent with TensorBoard logging (MLflow managed externally)"""

    # Initialize environment and agent
    env = Environment()
    agent = REINFORCEAgent()

    # Set up TensorBoard logging
    tb_log_dir = f"runs/reinforce_{int(time.time())}"
    writer = SummaryWriter(log_dir=tb_log_dir)

    for episode in range(n_episodes):
        states = []
        actions = []
        rewards = []

        state = env.reset()

        for step in range(max_steps):
            states.append(state)
            action = agent.sample_action(state)
            actions.append(action)

            reward = env.reward(state)
            rewards.append(reward)

            state = env.transition(state, action)

        returns = compute_returns(rewards, gamma)
        returns = apply_baseline(returns)
        agent.update_policy(states, actions, returns)

        total_reward = sum(rewards)
        mean_reward = np.mean(rewards)
        policy_std = np.exp(agent.log_std)

        # Store for in-code inspection
        agent.episode_rewards.append(total_reward)
        agent.policy_params.append([agent.w1, agent.b1, agent.log_std])

        if episode % visualize_every == 0:
            writer.add_scalar("Reward/Total", total_reward, episode)
            writer.add_scalar("Reward/Mean", mean_reward, episode)
            writer.add_scalar("Policy/LogStd", agent.log_std, episode)
            writer.add_scalar("Policy/Std", policy_std, episode)
            writer.flush()

        #     MLflow logging (using the active run from main)
        #     it is very slow! I decided to use tensorboard for logging during the run
        #     mlflow.log_metric("reward_total", total_reward, step=episode)
        #     mlflow.log_metric("reward_mean", mean_reward, step=episode)
        #     mlflow.log_metric("policy_std", policy_std, step=episode)
        #     # TensorBoard logging


    # Save TensorBoard logs as MLflow artifact
    writer.flush()
    writer.close()
    mlflow.log_artifacts(tb_log_dir, artifact_path="tensorboard_logs")

    return env, agent
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
def test_learned_policy(env, agent):
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
    plt.show()

def main():
    """Main training and evaluation loop"""
    print("Starting REINFORCE training...")
    print("Key fixes for large action spaces:")
    print("1. Smaller initial weights (w1 = 0.01 instead of 0.1)")
    print("2. Lower learning rates for stability")
    print("3. Baseline subtraction to reduce variance")
    print("4. Gradient clipping to prevent instability")
    print("5. Bounds on log_std to prevent extreme values")
    print("6. Increased initial exploration (higher initial std)")
    print("\nTraining in progress...")

    # Set MLflow experiment and start a single run
    n_episodes = 5000
    visualize_every=10
    # env, agent = train_reinforce_plot(n_episodes=n_episodes, visualize_every=visualize_every)

    mlflow.set_experiment("REINFORCE_1D_Gaussian_Control")
    
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("learning_rate_mean", 0.005)
        mlflow.log_param("learning_rate_std", 0.0005)
        mlflow.log_param("n_episodes", n_episodes)
        mlflow.log_param("gamma", 0.99)
        mlflow.log_param("max_steps", 20)

        # Train agent (this will use the active MLflow run)
    
        env, agent = train_reinforce(n_episodes=n_episodes, visualize_every=visualize_every)

        # Log final policy parameters
        mlflow.log_metric("final_w1", agent.w1)
        mlflow.log_metric("final_b1", agent.b1)
        mlflow.log_metric("final_std", agent.policy_std())

        # Log final average reward over last 50 episodes
        if len(agent.episode_rewards) >= 50:
            avg_reward = np.mean(agent.episode_rewards[-50:])
            mlflow.log_metric("final_avg_reward", avg_reward)

        print("\nTraining complete. Results logged to MLflow.")

    # Test policy (visualize) - outside the MLflow run context
    # test_learned_policy(env, agent)
        
    print("\nThese changes allow learning in the realistic [-10,10] action space!")
    print("The agent learns to navigate toward the reward peak at state=2.0")

if __name__ == "__main__":
    main()