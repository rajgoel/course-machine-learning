"""
# REINFORCE Algorithm Implementation

REINFORCE is a policy gradient method that directly optimizes the policy by following 
the gradient of expected reward. Key differences from DQN:

1. **Policy-based**: Learns a policy π(a|s) directly instead of value function Q(s,a)
2. **Stochastic**: Outputs action probabilities, not deterministic actions
3. **Monte Carlo**: Uses complete episode returns, no bootstrapping
4. **Policy Gradient**: Updates policy parameters using gradient ascent

Algorithm:
1. Initialize policy network π(a|s; θ)
2. For each episode:
   - Generate episode using current policy
   - Calculate returns G_t for each step t
   - Update policy: θ ← θ + α * ∇_θ log π(a_t|s_t) * G_t
"""

using Flux
using Statistics
import CommonRLInterface as RL
using ..Lecture08: EpisodeLogger


"""
    compute_returns(rewards, γ=0.99)

Compute discounted returns G_t = Σ(γ^k * r_{t+k}) for each time step.
This is the Monte Carlo return used in REINFORCE.
"""
function compute_returns(rewards, γ=0.99)
    T = length(rewards)
    returns = zeros(Float32, T)
    G = 0.0f0
    
    # Compute returns backwards from end of episode
    for t in T:-1:1
        G = rewards[t] + γ * G
        returns[t] = G
    end
    
    return returns
end

"""
    REINFORCE(env; kwargs...) -> Flux.Chain

Train an agent using the REINFORCE policy gradient algorithm.

# Arguments
- `env`: Environment implementing CommonRLInterface
- `hidden_layers=[128, 64]`: Architecture of hidden layers
- `η=1e-3`: Learning rate (typically higher than DQN)
- `γ=0.99`: Discount factor for returns
- `T=20_000`: Maximum steps per episode
- `max_episodes=1000`: Maximum number of episodes to train
- `callback=EpisodeLogger()`: Function called after each episode

# Returns
- Trained policy network (Flux.Chain)

# Example
```julia
env = BreakoutEnv()
policy = REINFORCE(env, max_episodes=1000)
```
"""
function REINFORCE(env; 
    hidden_layers=[64, 32], η=1e-3, 
    γ=0.99, T=20_000, max_episodes=1000, 
    callback=EpisodeLogger()
)
    
    # Initialize environment and get dimensions
    RL.reset!(env)
    n_actions = length(RL.actions(env))
    feature_dim = length(RL.observe(env))
    
    # Create policy network for discrete actions
    layers = [feature_dim, hidden_layers..., n_actions]
    policy = PolicyNetwork(layers)
    
    # Set up optimizer (Adam works well for policy gradients)
    optimizer = Flux.setup(Adam(η), policy)
    
    # Training loop
    for episode in 1:max_episodes
        # Reset environment
        RL.reset!(env)
        
        # Episode storage
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        # Generate episode
        total_reward = 0.0
        steps = 0
        
        while !RL.terminated(env)
            state = RL.observe(env)
            push!(states, copy(state))
            
            # Get action probabilities from policy
            logits = policy(reshape(state, :, 1))[:, 1]
            probs = Flux.softmax(logits)
            
            # Sample action from probability distribution
            action_idx = sample_action(probs)
            action = RL.actions(env)[action_idx]
            
            # Store log probability for gradient computation
            push!(log_probs, log(probs[action_idx]))
            push!(actions, action_idx)
            
            # Execute action
            reward = RL.act!(env, action)
            push!(rewards, reward)
            total_reward += reward
            steps += 1
        end
        
        # Compute advantages
        advantages = compute_returns(rewards, γ)
                
        # Policy gradient update
        policy_loss, policy_grads = Flux.withgradient(policy) do model
            # Compute policy loss: -∇_θ Σ log π(a|s) * G
            loss = 0.0f0
            for t in 1:length(states)
                logits = model(reshape(states[t], :, 1))[:, 1]
                log_probs = Flux.logsoftmax(logits)
                loss -= log_probs[actions[t]] * advantages[t]
            end
            loss / length(states)  # Average over episode
        end
        
        # Update policy parameters
        Flux.update!(optimizer, policy, policy_grads[1])
        
        # Callback for logging
        if callback !== nothing
            callback(episode, steps, total_reward)
        end
    end
    
    return policy
end


