""""
================
Experiment 1.
================
Environment: 1D gaussian.
Algorithm(s):
    1. REINFORCE with Gaussian policy and baseline.
    2. REINFORCE with Gaussian policy without baseline.
Episodes: 5000.
Steps per episode: 20.
Runs: 30

"""
from statistics import mean
from tqdm import tqdm
from pathlib import Path
from RLGaussianLibrary import REINFORCEAgent, Environment, train_agent
import mlflow
import numpy as np
from torch.utils.tensorboard import SummaryWriter
def main():
    experiment = mlflow.set_experiment("REINFORCE_1D_Gaussian_Control_baseline_vs_no_baseline")
    max_runs = 30
    n_episodes = 5000
    visualize_every=10
    lr_mean = 0.005
    lr_std = 0.0005
    gamma = 0.99
    max_steps = 20
    with mlflow.start_run() as run:
        mlflow.log_param("learning_rate_mean", lr_mean)
        mlflow.log_param("learning_rate_std", lr_std)
        mlflow.log_param("n_episodes", n_episodes)
        mlflow.log_param("gamma", gamma)
        mlflow.log_param("max_steps", max_steps)
        run_name = run.info.run_name
        print(f"MLflow run started with ID: {run_name}")
        #set tensorboard
        for baseline in [True, False]:
            tensorboard_dir = Path("runs", experiment.name, f"reinforce_{run_name}_baseline_{baseline}")
            writer = SummaryWriter(log_dir=str(tensorboard_dir))
            print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
            for run in tqdm(range(max_runs)): 
                history_log_std =[]
                history_std =[]
                history_std =[]
                history_b1 = []
                history_w1 = []
                history_mean_reward = []
                history_total_reward = []
                history_reward_last_50_episodes = []
                # Train agent
                env = Environment()
                agent = REINFORCEAgent()
                _, _, returns, rewards = train_agent(experiment.name, run_name, env, agent, n_episodes=n_episodes, log_tensorboard=False, baseline=baseline)
                
                # Add final policy parameters for this run to the lists
                history_w1.append(agent.w1)
                history_b1.append(agent.b1)
                history_log_std.append(agent.policy_std())
                history_std.append(np.exp(agent.policy_std())) 
                # Add final reward statistics to the lists
                history_total_reward.append(sum(rewards))
                history_mean_reward.append(np.mean(rewards))

                #log everything to tensorboard
                writer.add_scalar("Policy/w1", agent.w1, run)
                writer.add_scalar("Policy/b1", agent.b1, run)
                writer.add_scalar("Policy/log_std", agent.policy_std(), run)
                writer.add_scalar("Policy/std", np.exp(agent.policy_std()), run)
                writer.add_scalar("Reward/Total", sum(rewards), run)
                writer.add_scalar("Reward/Mean", np.mean(rewards), run)
                if len(agent.episode_rewards) >= 50:
                    avg_reward = np.mean(agent.episode_rewards[-50:])
                    history_reward_last_50_episodes.append(avg_reward)
                    writer.add_scalar("Reward/Mean[-50:]", avg_reward, run)
                writer.flush()
            # Log mean policy parameters
            mlflow.log_metric("final_w1",  mean(history_w1))
            mlflow.log_metric("final_b1",  mean(history_b1))
            mlflow.log_metric("final_log_std", mean(history_log_std))
            mlflow.log_metric("final_std", mean(history_std))
            mlflow.log_metric("Total reward average", mean(history_total_reward))
            mlflow.log_metric("Mean reward average", mean(history_mean_reward))
            
            # with tempfile.NamedTemporaryFile(prefix=f"reinforce_{run_name}_baseline_{baseline}_", suffix=".txt", delete=True) as tmp:
            fig = test_learned_policy_plot(env, agent, show=False)
            mlflow.log_figure(fig, f"example_reinforce_lr_std_{agent.lr_mean}_lr_std_{agent.lr_mean}_baseline_{baseline}_policy_plot.png")
if __name__ == "__main__":
    main()