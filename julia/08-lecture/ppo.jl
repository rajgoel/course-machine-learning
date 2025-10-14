"""
# Proximal Policy Optimization (PPO) Implementation

This module implements the Proximal Policy Optimization algorithm as described in the paper:
"Proximal Policy Optimization Algorithms" (Schulman et al., 2017)

Key concepts demonstrated:
- Actor-Critic architecture with policy and value networks
- Clipped surrogate objective for policy updates
- Advantage estimation using Generalized Advantage Estimation (GAE)
- Mini-batch updates with multiple epochs per batch
- Entropy regularization for exploration
"""

using Flux
using Statistics
using Distributions
import CommonRLInterface as RL

"""
    PPOActor(input_dim, hidden_layers, n_actions)

Actor network that outputs action probabilities.
Uses softmax activation for discrete action spaces.
"""
function PPOActor(input_dim::Int, hidden_layers::Vector{Int}, n_actions::Int)
    layers = []
    dims = [input_dim, hidden_layers..., n_actions]
    
    for i in 1:length(dims)-2
        push!(layers, Flux.Dense(dims[i], dims[i+1], Flux.tanh))
    end
    push!(layers, Flux.Dense(dims[end-1], dims[end]))
    push!(layers, Flux.softmax)
    
    return Flux.Chain(layers...)
end

"""
    PPOCritic(input_dim, hidden_layers)

Critic network that outputs state values.
Single output neuron with no activation function.
"""
function PPOCritic(input_dim::Int, hidden_layers::Vector{Int})
    layers = []
    dims = [input_dim, hidden_layers..., 1]
    
    for i in 1:length(dims)-2
        push!(layers, Flux.Dense(dims[i], dims[i+1], Flux.tanh))
    end
    push!(layers, Flux.Dense(dims[end-1], dims[end]))
    
    return Flux.Chain(layers...)
end

"""
    PPOExperience

Stores experience for PPO training including log probabilities and advantages.
"""
mutable struct PPOExperience
    states::Vector{Vector{Float32}}
    actions::Vector{Int}
    rewards::Vector{Float32}
    values::Vector{Float32}
    log_probs::Vector{Float32}
    terminals::Vector{Bool}
    advantages::Vector{Float32}
    returns::Vector{Float32}
    
    function PPOExperience()
        new(Vector{Float32}[], Int[], Float32[], Float32[], Float32[], Bool[], Float32[], Float32[])
    end
end

"""
    reset!(experience::PPOExperience)

Clear all stored experience data.
"""
function reset!(experience::PPOExperience)
    empty!(experience.states)
    empty!(experience.actions)
    empty!(experience.rewards)
    empty!(experience.values)
    empty!(experience.log_probs)
    empty!(experience.terminals)
    empty!(experience.advantages)
    empty!(experience.returns)
end

"""
    compute_advantages!(experience, gamma, lambda)

Compute advantages and returns using Generalized Advantage Estimation (GAE).
"""
function compute_advantages!(experience::PPOExperience, gamma::Float32=0.99f0, lambda::Float32=0.95f0)
    n = length(experience.rewards)
    experience.advantages = zeros(Float32, n)
    experience.returns = zeros(Float32, n)
    
    last_advantage = 0.0f0
    
    for t in reverse(1:n)
        if t == n
            next_value = experience.terminals[t] ? 0.0f0 : experience.values[t]
        else
            next_value = experience.values[t+1]
        end
        
        delta = experience.rewards[t] + gamma * next_value - experience.values[t]
        last_advantage = delta + gamma * lambda * last_advantage * (1 - experience.terminals[t])
        experience.advantages[t] = last_advantage
    end
    
    experience.returns = experience.advantages + experience.values
    
    # Normalize advantages
    if length(experience.advantages) > 1
        mean_adv = mean(experience.advantages)
        std_adv = std(experience.advantages)
        if std_adv > 1e-8
            experience.advantages .= (experience.advantages .- mean_adv) ./ std_adv
        end
    end
end

"""
    select_action(actor, state)

Select action using the actor network and return action, log probability, and action probabilities.
"""
function select_action(actor, state::Vector{Float32})
    state_matrix = reshape(state, :, 1)
    probs = actor(state_matrix)[:, 1]
    
    # Sample action from categorical distribution
    dist = Categorical(probs)
    action = rand(dist)
    log_prob = log(probs[action] + 1e-8)  # Add small epsilon to avoid log(0)
    
    return action, log_prob, probs
end

"""
    ppo_loss(actor, critic, experience, batch_indices, clip_ratio, value_coef, entropy_coef)

Compute PPO loss including policy loss, value loss, and entropy regularization.
"""
function ppo_loss(actor, critic, experience::PPOExperience, batch_indices::Vector{Int}, 
                  clip_ratio::Float32=0.2f0, value_coef::Float32=0.5f0, entropy_coef::Float32=0.01f0)
    
    batch_size = length(batch_indices)
    states_batch = reduce(hcat, experience.states[batch_indices])
    actions_batch = experience.actions[batch_indices]
    old_log_probs_batch = experience.log_probs[batch_indices]
    advantages_batch = experience.advantages[batch_indices]
    returns_batch = experience.returns[batch_indices]
    
    # Forward pass through networks
    current_probs = actor(states_batch)
    current_values = vec(critic(states_batch))
    
    # Compute new log probabilities
    current_log_probs = [log(current_probs[actions_batch[i], i] + 1e-8) for i in 1:batch_size]
    
    # Policy loss (clipped surrogate objective)
    ratio = exp.(current_log_probs - old_log_probs_batch)
    clipped_ratio = clamp.(ratio, 1 - clip_ratio, 1 + clip_ratio)
    policy_loss = -mean(min.(ratio .* advantages_batch, clipped_ratio .* advantages_batch))
    
    # Value loss (MSE between predicted values and returns)
    value_loss = mean((current_values - returns_batch).^2)
    
    # Entropy loss (encourage exploration)
    entropy = -sum(current_probs .* log.(current_probs .+ 1e-8), dims=1)
    entropy_loss = -mean(entropy)
    
    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    
    return total_loss, policy_loss, value_loss, entropy_loss
end

"""
    train_step!(actor, critic, actor_opt, critic_opt, experience, batch_size, epochs, clip_ratio, value_coef, entropy_coef)

Perform multiple epochs of mini-batch training on collected experience.
"""
function train_step!(actor, critic, actor_opt, critic_opt, experience::PPOExperience, 
                     batch_size::Int=64, epochs::Int=4, clip_ratio::Float32=0.2f0, 
                     value_coef::Float32=0.5f0, entropy_coef::Float32=0.01f0)
    
    n_samples = length(experience.states)
    total_policy_loss = 0.0f0
    total_value_loss = 0.0f0
    total_entropy_loss = 0.0f0
    n_updates = 0
    
    for epoch in 1:epochs
        # Shuffle indices for each epoch
        indices = randperm(n_samples)
        
        for start_idx in 1:batch_size:n_samples
            end_idx = min(start_idx + batch_size - 1, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            if length(batch_indices) < 2  # Skip tiny batches
                continue
            end
            
            # Update actor
            loss, grads = Flux.withgradient(actor) do model
                total_loss, policy_loss, value_loss, entropy_loss = ppo_loss(
                    model, critic, experience, batch_indices, clip_ratio, value_coef, entropy_coef)
                total_policy_loss += policy_loss
                total_entropy_loss += entropy_loss
                return total_loss
            end
            
            Flux.update!(actor_opt, actor, grads[1])
            
            # Update critic
            critic_loss, critic_grads = Flux.withgradient(critic) do model
                states_batch = reduce(hcat, experience.states[batch_indices])
                returns_batch = experience.returns[batch_indices]
                current_values = vec(model(states_batch))
                return mean((current_values - returns_batch).^2)
            end
            
            Flux.update!(critic_opt, critic, critic_grads[1])
            total_value_loss += critic_loss
            n_updates += 1
        end
    end
    
    avg_policy_loss = n_updates > 0 ? total_policy_loss / n_updates : 0.0f0
    avg_value_loss = n_updates > 0 ? total_value_loss / n_updates : 0.0f0
    avg_entropy_loss = n_updates > 0 ? total_entropy_loss / n_updates : 0.0f0
    
    return avg_policy_loss, avg_value_loss, avg_entropy_loss
end

"""
    PPO(env; kwargs...) -> (actor, critic)

Train a PPO agent using the algorithm from Schulman et al. 2017.

# Arguments
- `env`: Environment implementing CommonRLInterface
- `max_episodes=100_000`: Maximum number of episodes to train
- `max_steps_per_episode=20000`: Maximum steps per episode
- `batch_size=64`: Mini-batch size for training
- `learning_rate=3e-4`: Learning rate for both actor and critic
- `gamma=0.99`: Discount factor for future rewards
- `lambda=0.95`: GAE lambda parameter
- `clip_ratio=0.2`: PPO clipping parameter
- `value_coef=0.5`: Value loss coefficient
- `entropy_coef=0.01`: Entropy regularization coefficient
- `update_frequency=2048`: Collect this many steps before updating
- `epochs=4`: Number of epochs per update
- `hidden_layers=[128, 64]`: Architecture of hidden layers
- `callback=EpisodeLogger()`: Function called after each episode

# Returns
- Tuple of trained (actor, critic) networks
"""
function PPO(env; max_episodes=100_000, max_steps_per_episode=20000, batch_size=64,
             learning_rate=3e-4, gamma=0.99f0, lambda=0.95f0, clip_ratio=0.2f0,
             value_coef=0.5f0, entropy_coef=0.01f0, update_frequency=2048, epochs=4,
             hidden_layers=[128, 64], callback=EpisodeLogger())
    
    # Initialize environment and get dimensions
    n_actions = length(RL.actions(env))
    RL.reset!(env)
    initial_state = RL.observe(env)
    feature_dim = length(initial_state)
    
    println("PPO Training Configuration:")
    println("  Actions: ", RL.actions(env))
    println("  Feature dimension: ", feature_dim)
    println("  Actor architecture: ", [feature_dim, hidden_layers..., n_actions])
    println("  Critic architecture: ", [feature_dim, hidden_layers..., 1])
    println("  Update frequency: ", update_frequency)
    println("  Batch size: ", batch_size)
    println("  Epochs per update: ", epochs)
    
    # Initialize networks
    actor = PPOActor(feature_dim, hidden_layers, n_actions)
    critic = PPOCritic(feature_dim, hidden_layers)
    
    # Initialize optimizers
    actor_opt = Flux.setup(Adam(learning_rate), actor)
    critic_opt = Flux.setup(Adam(learning_rate), critic)
    
    # Initialize experience buffer
    experience = PPOExperience()
    
    total_steps = 0
    episode_count = 0
    
    while episode_count < max_episodes
        # Collect experience
        steps_collected = 0
        
        while steps_collected < update_frequency && episode_count < max_episodes
            episode_count += 1
            RL.reset!(env)
            state = RL.observe(env)
            episode_reward = 0.0
            
            for step in 1:max_steps_per_episode
                total_steps += 1
                steps_collected += 1
                
                # Select action using current policy
                action_idx, log_prob, _ = select_action(actor, state)
                action = RL.actions(env)[action_idx]
                
                # Get value estimate
                value = critic(reshape(state, :, 1))[1]
                
                # Execute action
                reward = RL.act!(env, action)
                next_state = RL.observe(env)
                terminal = RL.terminated(env)
                
                # Store experience
                push!(experience.states, copy(state))
                push!(experience.actions, action_idx)
                push!(experience.rewards, Float32(reward))
                push!(experience.values, value)
                push!(experience.log_probs, log_prob)
                push!(experience.terminals, terminal)
                
                episode_reward += reward
                state = next_state
                
                if terminal || steps_collected >= update_frequency
                    break
                end
            end
            
            # Log episode completion
            callback(episode_count, total_steps, episode_reward, 0.0)  # PPO doesn't use epsilon
            
            if steps_collected >= update_frequency
                break
            end
        end
        
        # Compute advantages and returns
        compute_advantages!(experience, gamma, lambda)
        
        # Perform training updates
        if length(experience.states) >= batch_size
            policy_loss, value_loss, entropy_loss = train_step!(
                actor, critic, actor_opt, critic_opt, experience, 
                batch_size, epochs, clip_ratio, value_coef, entropy_coef)
            
            if episode_count % 10 == 0
                println("  Policy Loss: $(round(policy_loss, digits=4)), " *
                       "Value Loss: $(round(value_loss, digits=4)), " *
                       "Entropy Loss: $(round(entropy_loss, digits=4))")
            end
        end
        
        # Clear experience buffer
        reset!(experience)
    end
    
    return actor, critic
end
