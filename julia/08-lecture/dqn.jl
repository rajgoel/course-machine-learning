"""
# Deep Q-Network (DQN) Implementation

This module implements the Deep Q-Network algorithm as described in the DeepMind paper:
"Human-level control through deep reinforcement learning" (Nature, 2015)

Key concepts demonstrated:
- Experience replay for breaking temporal correlations
- Target networks for stable Q-learning
- Epsilon-greedy exploration strategy
- Neural network function approximation
"""

using Flux
using Statistics
import CommonRLInterface as RL

"""
    update_target_network!(target_network, source_network)

Copy weights from the main Q-network to the target network.

The target network is a key innovation in DQN that provides stable
Q-learning targets. Without it, the Q-learning updates would be chasing
a moving target (since both the current Q-values and target Q-values
come from the same network), leading to instability.

# Arguments
- `target_network`: The target network to update
- `source_network`: The main Q-network to copy weights from
"""
function update_target_network!(target_network, source_network)
    # Get the current state (weights and biases) of both networks
    target_state = Flux.state(target_network)
    source_state = Flux.state(source_network)
    
    # Copy the source network's weights to the target network
    Flux.loadmodel!(target_network, source_state)
end

"""
    train_step!(q_network, target_network, optimizer, replay_buffer, batch_size, discount_factor, env_actions) -> Float32

Perform one training step of the DQN algorithm.

This implements the core Q-learning update using neural networks:
1. Sample a batch of experiences from replay buffer
2. Compute target Q-values using target network: y = r + γ * max(Q_target(s'))
3. Compute current Q-values using main network: Q(s,a)
4. Compute loss (Huber loss) between target and current Q-values
5. Update the main network using backpropagation

# Arguments
- `q_network`: Main Q-network being trained
- `target_network`: Target network for stable Q-learning targets
- `optimizer`: Flux optimizer (typically Adam)
- `replay_buffer`: Buffer of past experiences
- `batch_size`: Number of experiences to learn from
- `discount_factor`: γ (gamma) - how much to value future rewards
- `env_actions`: Available actions in the environment

# Returns
- Loss value for monitoring training progress
"""
function train_step!(q_network, target_network, optimizer, replay_buffer, batch_size, discount_factor, env_actions)
    # Sample a random batch of transitions from experience replay
    batch = sample(replay_buffer, batch_size)
    
    # Extract components from the batch for efficient processing
    # These helper functions convert the batch into matrix form
    states() = reduce(hcat, getfield.(batch, :state))  # (features, batch_size)
    action_indices() = [findfirst(==(action), env_actions) for action in getfield.(batch, :action)]
    next_states() = reduce(hcat, getfield.(batch, :next_state))  # (features, batch_size)
    rewards() = getfield.(batch, :reward)  # (batch_size,)
    terminal_values() = Float32.(getfield.(batch, :terminal))  # (batch_size,)
    
    # Compute target Q-values using the target network (key DQN innovation)
    # Target network provides stable targets, preventing moving target problem
    target_q_values = target_network(next_states())  # (n_actions, batch_size)
    max_q_values = maximum(target_q_values, dims=1)[:]  # Max Q-value for each next state
    
    # Bellman equation: Q*(s,a) = r + γ * max_a' Q*(s',a')
    # If episode terminated, future reward is 0
    y = rewards() + (1 .- terminal_values()) .* discount_factor .* max_q_values
        
    # Compute current Q-values and loss using automatic differentiation
    loss, grads = Flux.withgradient(q_network) do model
        Q = model(states())  # (n_actions, batch_size)
        # Extract Q-values for the actions that were actually taken
        current_q_value = [Q[action_indices()[i], i] for i in 1:batch_size]
        # Use Huber loss (less sensitive to outliers than MSE)
        Flux.Losses.huber_loss(y, current_q_value)
    end
    
    # Update the network weights using the computed gradients
    Flux.update!(optimizer, q_network, grads[1])
    
    return loss
end

"""
    DQN(env; kwargs...) -> Flux.Chain

Train a Deep Q-Network agent using the algorithm from the DeepMind Nature paper.

This function implements the complete DQN training loop with all the key
components that made DQN successful:

1. **Experience Replay**: Store and randomly sample past experiences
2. **Target Networks**: Use a separate network for computing targets
3. **Epsilon-greedy Exploration**: Balance exploration vs exploitation
4. **Neural Network Function Approximation**: Use deep networks for Q-values

# Arguments
- `env`: Environment implementing CommonRLInterface
- `max_episodes=100_000`: Maximum number of episodes to train
- `max_steps_per_episode=20_000`: Maximum steps per episode
- `replay_memory_size=1_000_000`: Size of experience replay buffer
- `update_frequency=4`: How often to perform training steps
- `target_update_frequency=10_000`: How often to update target network
- `batch_size=32`: Batch size for training
- `learning_rate=1e-4`: Learning rate for Adam optimizer
- `discount_factor=0.99`: Gamma (γ) - discount factor for future rewards
- `replay_start_size=100_000`: Start training after this many experiences
- `initial_exploration=1`: Starting epsilon for exploration
- `final_exploration=0.01`: Final epsilon (minimum exploration)
- `exploration_steps=5_000_000`: Steps over which to anneal epsilon
- `hidden_layers=[128, 64]`: Architecture of hidden layers
- `callback=EpisodeLogger()`: Function called after each episode

# Returns
- Trained Q-network (Flux.Chain)

# Examples

## Basic usage:
```julia
env = BreakoutEnv()
q_network = DQN(env)
```

## Turn off target network (for comparison studies):
```julia
env = BreakoutEnv()
q_network = DQN(env,
               target_update_frequency=1  # Update target every step = no target network
            )
```

## Vanilla Q-learning (no experience replay, no target network):
```julia
env = BreakoutEnv()
q_network = DQN(env,
               target_update_frequency=1,    # No target network
               update_frequency=1,           # Update every step
               batch_size=1,                 # No batching
               replay_start_size=0          # Start training immediately
            )
```


"""
function DQN(env; max_episodes=100_000, max_steps_per_episode=20_000, replay_memory_size=1_000_000, 
                                  update_frequency=4, target_update_frequency=10_000, batch_size=32, learning_rate=1e-4, discount_factor=0.99, 
                                  replay_start_size=100_000, initial_exploration=1, final_exploration=0.01,
                                  exploration_steps=5_000_000, hidden_layers=[128, 64], callback=EpisodeLogger())
    
    # Initialize environment and get dimensions
    n_actions = length(RL.actions(env))
    
    # Get feature dimension from environment state
    RL.reset!(env)
    initial_state = RL.observe(env)
    feature_dim = length(initial_state)
    
    # Print training configuration for educational purposes
    println("Environment initialized:")
    println("  Actions: ", RL.actions(env))
    println("  Feature dimension: ", feature_dim)
    println("  Max episodes: ", max_episodes)
    println("  Max steps per episode: ", max_steps_per_episode)
    println("  Network architecture: ", [feature_dim, hidden_layers..., n_actions])
    
    # Initialize experience replay buffer
    replay_buffer = ReplayBuffer(replay_memory_size)
    
    # Create the neural network architecture
    # Input: flattened game state, Output: Q-values for each action
    layers = [feature_dim, hidden_layers..., n_actions]
    q_network = QNetwork(layers)        # Main Q-network
    target_network = QNetwork(layers)   # Target network for stable learning
    
    # Initialize target network with same weights as main network
    # This ensures they start identical before diverging during training
    update_target_network!(target_network, q_network)
    
    # Set up Adam optimizer for gradient-based learning
    optimizer = Flux.setup(Adam(learning_rate), q_network)
    
    total_steps = 0  # Global step counter across all episodes
    
    # Main training loop: Episodes
    for episode in 1:max_episodes
        # Reset environment for new episode
        RL.reset!(env)
        state = RL.observe(env)
        episode_reward = 0.0
        epsilon = initial_exploration  # Will be updated during episode
        
        # Episode loop: Steps within current episode
        for step in 1:max_steps_per_episode
            total_steps += 1
            
            # Epsilon-greedy action selection (exploration vs exploitation)
            # Linearly anneal epsilon from initial to final over exploration_steps
            epsilon = max((final_exploration - initial_exploration) / exploration_steps * total_steps + initial_exploration,
                          final_exploration)
            
            if rand() < epsilon
                # Exploration: Random action to discover new strategies
                action_idx = rand(1:n_actions)
                action = RL.actions(env)[action_idx]
            else
                # Exploitation: Greedy action based on current Q-network
                q_values = q_network(reshape(state, :, 1))
                action_idx = argmax(q_values[:, 1])  # Action with highest Q-value
                action = RL.actions(env)[action_idx]
            end
            
            # Execute the chosen action in the environment
            reward = RL.act!(env, action)
            next_state = RL.observe(env)
            terminal = RL.terminated(env)
            
            # Store experience in replay buffer for later learning
            push!(replay_buffer, StateTransition(state, action, Float32(reward), next_state, terminal))
            
            # Update episode statistics
            episode_reward += reward
            state = next_state
            
            # Perform training step (only after sufficient experience collected)
            if total_steps > replay_start_size && total_steps % update_frequency == 0
                loss = train_step!(q_network, target_network, optimizer, replay_buffer, batch_size, discount_factor, RL.actions(env))
            end
            
            # Periodically update target network (key for stability)
            if total_steps > replay_start_size && total_steps % target_update_frequency == 0
                update_target_network!(target_network, q_network)
            end
            
            # End episode if terminal state reached
            if terminal
                break
            end
        end
        
        # Log episode completion using callback (for monitoring progress)
        callback(episode, total_steps, episode_reward, epsilon)
    end
    
    return q_network
end

