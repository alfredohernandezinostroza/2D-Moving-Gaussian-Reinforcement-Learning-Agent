%% REINFORCE Algorithm Demo - 1D Continuous Control (Fixed for Large Action Space)
% This demo shows how policy gradient methods work using REINFORCE
% on a simple 1D problem where the agent must find the peak of a Gaussian reward
% FIXED: Proper handling of large action spaces [-10, 10]

clear; close all; clc;

%% Problem Setup
% Environment: 1D continuous state space [-5, 5]
% Action: 1D continuous action that moves the agent (realistic human clicking)
% Reward: Gaussian centered at state = 2.0 with std = 1.0
% Goal: Learn a policy that navigates to the peak reward

% Environment parameters
state_min = -5;
state_max = 5;
action_min = -10;  % Realistic: click from 5 to -5
action_max = 10;   % Realistic: click from -5 to 5
reward_center = 2.0;  % Peak of Gaussian reward
reward_std = 1.0;

% Reward function
reward_fn = @(state) exp(-0.5 * ((state - reward_center) / reward_std)^2);

% State transition (deterministic for simplicity)
transition_fn = @(state, action) max(state_min, min(state_max, state + action));

%% Policy Network Setup - FIXED VERSION
% We use a simple Gaussian policy: action ~ N(mu(s), sigma)
% KEY FIX: Proper scaling and initialization for large action space

% Policy parameters (what we'll learn)
% IMPORTANT: Scale initial weights down when action space is large
w1 = 0.01;  % Reduced from 0.1 - smaller initial weights for large action space
b1 = 0.0;   % Bias
log_std = log(2.0);  % Increased initial exploration for large action space

% Learning rates - ADJUSTED for larger action space
lr_mean = 0.005;   % Reduced learning rate for stability
lr_std = 0.0005;   % Reduced learning rate for std

% Training parameters
n_episodes = 5000;
max_steps = 20;
gamma = 0.99;  % Discount factor

% Storage for results
episode_rewards = zeros(n_episodes, 1);
policy_params = zeros(n_episodes, 3);  % [w1, b1, log_std]

%% Visualization Setup
figure('Position', [100, 100, 1200, 800]);

%% REINFORCE Training Loop
for episode = 1:n_episodes
    % Storage for this episode
    states = [];
    actions = [];
    rewards = [];
    
    % Initial state (random)
    state = (state_max - state_min) * rand() + state_min;
    
    % Generate episode
    for step = 1:max_steps
        % Store state
        states = [states; state];
        
        % Compute policy mean - FIXED: Proper scaling
        % Use full action range scaling
        mu = action_max * tanh(w1 * state + b1);
        sigma = exp(log_std);
        
        % Sample action from policy
        action = mu + sigma * randn();
        % Clip action to valid range
        action = max(action_min, min(action_max, action));
        actions = [actions; action];
        
        % Get reward
        reward = reward_fn(state);
        rewards = [rewards; reward];
        
        % Transition to next state
        state = transition_fn(state, action);
    end
    
    % Compute returns (discounted cumulative rewards)
    returns = zeros(length(rewards), 1);
    G = 0;
    for t = length(rewards):-1:1
        G = rewards(t) + gamma * G;
        returns(t) = G;
    end
    
    % Baseline subtraction (improved stability for large action spaces)
    if ~isempty(returns) && length(returns) > 1
        baseline = mean(returns);  % Simple baseline
        returns = returns - baseline;
        % Optional: Also normalize
        if std(returns) > 1e-8
            returns = returns / std(returns);
        end
    elseif ~isempty(returns)
        returns = returns * 0;  % Single return, set to 0
    end
    
    % REINFORCE update
    grad_w1 = 0;
    grad_b1 = 0;
    grad_log_std = 0;
    
    if ~isempty(states)  % Only update if we have data
        for t = 1:length(states)
            % Recompute policy for this state-action pair
            state_t = states(t);
            action_t = actions(t);
            mu_t = action_max * tanh(w1 * state_t + b1);
            sigma_t = exp(log_std);
            
            % Compute log probability of action
            log_prob = -0.5 * log(2 * pi) - log_std - 0.5 * ((action_t - mu_t) / sigma_t)^2;
            
            % Compute gradients of log probability
            % d/d_mu log p(a|s) = (a - mu) / sigma^2
            d_log_prob_d_mu = (action_t - mu_t) / (sigma_t^2);
            
            % d/d_params mu(s) using chain rule
            tanh_term = tanh(w1 * state_t + b1);
            d_tanh = 1 - tanh_term^2;  % Derivative of tanh
            d_mu_d_w1 = action_max * d_tanh * state_t;
            d_mu_d_b1 = action_max * d_tanh;
            
            % Gradient of log_std
            d_log_prob_d_log_std = ((action_t - mu_t)^2 / sigma_t^2 - 1);
            
            % REINFORCE gradients: grad = return * grad_log_prob
            grad_w1 = grad_w1 + returns(t) * d_log_prob_d_mu * d_mu_d_w1;
            grad_b1 = grad_b1 + returns(t) * d_log_prob_d_mu * d_mu_d_b1;
            grad_log_std = grad_log_std + returns(t) * d_log_prob_d_log_std;
        end
        
        % Update parameters (gradient ascent) with gradient clipping
        grad_clip = 1.0;  % Gradient clipping for stability
        grad_w1 = max(-grad_clip, min(grad_clip, grad_w1 / length(states)));
        grad_b1 = max(-grad_clip, min(grad_clip, grad_b1 / length(states)));
        grad_log_std = max(-grad_clip, min(grad_clip, grad_log_std / length(states)));
        
        w1 = w1 + lr_mean * grad_w1;
        b1 = b1 + lr_mean * grad_b1;
        log_std = log_std + lr_std * grad_log_std;
        
        % Prevent log_std from becoming too extreme
        log_std = max(log(0.1), min(log(5.0), log_std));
    end
    
    % Store results
    episode_rewards(episode) = sum(rewards);
    policy_params(episode, :) = [w1, b1, log_std];
    
    % Visualization every 50 episodes for better performance
    if mod(episode, 50) == 0 || episode == 1
        clf;
        
        % Plot 1: Environment and Policy
        subplot(2, 2, 1);
        x_range = linspace(state_min, state_max, 100);
        r_range = arrayfun(reward_fn, x_range);
        plot(x_range, r_range, 'b-', 'LineWidth', 2);
        hold on;
        
        % Plot policy mean
        mu_range = action_max * tanh(w1 * x_range + b1);
        yyaxis right;
        plot(x_range, mu_range, 'r-', 'LineWidth', 2);
        ylabel('Policy Mean Action', 'Color', 'r');
        ylim([action_min, action_max]);
        
        yyaxis left;
        ylabel('Reward', 'Color', 'b');
        xlabel('State');
        title(sprintf('Environment and Learned Policy (Episode %d)', episode));
        grid on;
        legend('Reward Function', 'Policy Mean', 'Location', 'best');
        
        % Plot 2: Episode trajectory
        subplot(2, 2, 2);
        if ~isempty(states)
            plot(states, 'o-', 'LineWidth', 1.5);
            hold on;
            plot([1, length(states)], [reward_center, reward_center], 'r--');
        end
        xlabel('Step');
        ylabel('State');
        title('Last Episode Trajectory');
        legend('States', 'Optimal State', 'Location', 'best');
        grid on;
        
        % Plot 3: Learning curve
        subplot(2, 2, 3);
        plot(1:episode, episode_rewards(1:episode), 'LineWidth', 1.5);
        xlabel('Episode');
        ylabel('Total Reward');
        title('Learning Progress');
        grid on;
        
        % Plot 4: Parameter evolution
        subplot(2, 2, 4);
        plot(1:episode, policy_params(1:episode, 1), 'r-', 'LineWidth', 1.5);
        hold on;
        plot(1:episode, policy_params(1:episode, 2), 'g-', 'LineWidth', 1.5);
        plot(1:episode, exp(policy_params(1:episode, 3)), 'b-', 'LineWidth', 1.5);
        xlabel('Episode');
        ylabel('Parameter Value');
        title('Policy Parameters');
        legend('w1', 'b1', 'std', 'Location', 'best');
        grid on;
        
        drawnow;
    end
end

%% Final Evaluation
fprintf('\n=== REINFORCE Training Complete ===\n');
fprintf('Final parameters:\n');
fprintf('  w1 = %.3f\n', w1);
fprintf('  b1 = %.3f\n', b1);
fprintf('  std = %.3f\n', exp(log_std));
fprintf('\nFinal average reward: %.3f\n', mean(episode_rewards(max(1,end-50):end)));

% Test the learned policy
test_states = linspace(state_min, state_max, 5);
figure;
subplot(2, 2, [1 2]);
% Plot reward landscape
x_range = linspace(state_min, state_max, 100);
r_range = arrayfun(reward_fn, x_range);
plot(x_range, r_range, 'b-', 'LineWidth', 3);
hold on;

% Plot trajectories on the reward landscape
colors = lines(length(test_states));
for i = 1:length(test_states)
    state = test_states(i);
    trajectory = [state];
    
    for step = 1:10
        mu = action_max * tanh(w1 * state + b1);
        state = transition_fn(state, mu);
        trajectory = [trajectory, state];
    end
    
    % Plot trajectory as points on the 1D line at reward height
    for j = 1:length(trajectory)
        plot(trajectory(j), reward_fn(trajectory(j)), 'o', ...
             'Color', colors(i,:), 'MarkerSize', 8 - j/2, ...
             'MarkerFaceColor', colors(i,:));
    end
    % Connect with line
    plot(trajectory, arrayfun(reward_fn, trajectory), '-', ...
         'Color', colors(i,:), 'LineWidth', 2);
    
    % Add arrow showing direction
    if length(trajectory) > 1
        quiver(trajectory(1), reward_fn(trajectory(1)), ...
               trajectory(2)-trajectory(1), ...
               reward_fn(trajectory(2))-reward_fn(trajectory(1)), ...
               0.5, 'Color', colors(i,:), 'LineWidth', 2);
    end
end

plot([reward_center reward_center], [0 1], 'r--', 'LineWidth', 2);
xlabel('State');
ylabel('Reward');
title('Agent Trajectories on Reward Landscape');
legend('Reward Function', 'Optimal State', 'Location', 'best');
grid on;

% Plot temporal evolution
subplot(2, 2, 3);
for i = 1:length(test_states)
    state = test_states(i);
    trajectory = [state];
    
    for step = 1:10
        mu = action_max * tanh(w1 * state + b1);
        state = transition_fn(state, mu);
        trajectory = [trajectory, state];
    end
    
    plot(0:length(trajectory)-1, trajectory, 'o-', ...
         'Color', colors(i,:), 'LineWidth', 2);
    hold on;
end
plot([0 10], [reward_center reward_center], 'r--', 'LineWidth', 2);
xlabel('Time Step');
ylabel('State');
title('State Evolution Over Time');
grid on;

subplot(2, 2, 4);
% Show policy visualization
states_plot = linspace(state_min, state_max, 100);
mu_plot = action_max * tanh(w1 * states_plot + b1);
sigma_plot = exp(log_std) * ones(size(states_plot));

plot(states_plot, mu_plot, 'r-', 'LineWidth', 2);
hold on;
fill([states_plot, fliplr(states_plot)], ...
     [mu_plot + sigma_plot, fliplr(mu_plot - sigma_plot)], ...
     'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
xlabel('State');
ylabel('Action');
title('Learned Gaussian Policy');
legend('Mean', '±1 std', 'Location', 'best');
grid on;

%% Key Concepts Demonstrated
fprintf('\n=== Key Fixes for Large Action Spaces ===\n');
fprintf('1. Smaller initial weights (w1 = 0.01 instead of 0.1)\n');
fprintf('2. Lower learning rates for stability\n');
fprintf('3. Baseline subtraction to reduce variance\n');
fprintf('4. Gradient clipping to prevent instability\n');
fprintf('5. Bounds on log_std to prevent extreme values\n');
fprintf('6. Increased initial exploration (higher initial std)\n');
fprintf('\nThese changes allow learning in the realistic [-10,10] action space!\n');