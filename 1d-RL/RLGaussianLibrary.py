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
from pathlib import Path
from urllib.parse import quote
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
def train_agent(experiment_name, run_name, env, agent, n_episodes=5000, max_steps=20, gamma=0.99, visualize_every=1):
    """Train REINFORCE agent with TensorBoard logging (MLflow managed externally)"""

    # Set up TensorBoard logging
    if log_tensorboard:
        tensorboard_dir = Path("runs", experiment_name, f"reinforce_{run_name}")
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
        mlflow.set_tag("tensorboard_url", f"http://localhost:6006/#scalars&run={quote(run_name)}")

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
        if baseline:
            returns = apply_baseline(returns)
        agent.update_policy(states, actions, returns)

        total_reward = sum(rewards)
        mean_reward = np.mean(rewards)
        policy_std = np.exp(agent.log_std)

        # Store for in-code inspection
        agent.episode_rewards.append(total_reward)
        agent.policy_params.append([agent.w1, agent.b1, agent.log_std])

        if log_tensorboard and episode % visualize_every == 0:
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

    if log_tensorboard:
        # Save TensorBoard logs as MLflow artifact
        writer.flush()
        writer.close()
        mlflow.log_artifacts(tensorboard_dir, artifact_path="tensorboard_logs")

    return env, agent

def train_reinforce(experiment_name, run_name=None, n_episodes=5000, max_steps=20, gamma=0.99, visualize_every=1, baseline=True):
    """Train REINFORCE agent with TensorBoard logging (MLflow managed externally)"""

    # Initialize environment and agent
    env = Environment()
    agent = REINFORCEAgent()

    # Set up TensorBoard logging
    if run_name is None:
        run_name = f"time_id_{int(time.time())}"
    tensorboard_dir = Path("runs", experiment_name, f"reinforce_{run_name}")
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
    mlflow.set_tag("tensorboard_url", f"http://localhost:6006/#scalars&run={quote(run_name)}")

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
        if baseline:
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
    mlflow.log_artifacts(tensorboard_dir, artifact_path="tensorboard_logs")

    return env, agent
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

    experiment = mlflow.set_experiment("REINFORCE_1D_Gaussian_Control")
    
    with mlflow.start_run() as run:
        run_name = run.info.run_name
        print(f"MLflow run started with ID: {run_name}")
        # Log hyperparameters
        mlflow.log_param("learning_rate_mean", 0.005)
        mlflow.log_param("learning_rate_std", 0.0005)
        mlflow.log_param("n_episodes", n_episodes)
        mlflow.log_param("gamma", 0.99)
        mlflow.log_param("max_steps", 20)

        # Train agent (this will use the active MLflow run)
        env = Environment()
        agent = REINFORCEAgent()
        train_agent(experiment.name, run_name, env, agent, n_episodes=n_episodes, visualize_every=visualize_every)
        # env, agent = train_reinforce(experiment.name,run_name=run_name, n_episodes=n_episodes, visualize_every=visualize_every)

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