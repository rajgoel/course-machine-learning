using Flux
using Statistics
import CommonRLInterface as RL

"""
    dqn_target_evaluation(q_network, target_network, next_states, target_q_values, batch_size)

Standard DQN target evaluation: use target network for both action selection and evaluation.
"""
function dqn_target_evaluation(q_network, target_network, next_states, target_q_values, batch_size)
    return maximum(target_q_values, dims=1)[:]  # Max Q-value for each next state
end

"""
    ddqn_target_evaluation(q_network, target_network, next_states, target_q_values, batch_size)

Double DQN target evaluation: use main network to select actions, target network to evaluate them.
This reduces overestimation bias in Q-learning.
"""
function ddqn_target_evaluation(q_network, target_network, next_states, target_q_values, batch_size)
    q_values = q_network(next_states)  # (n_actions, batch_size)
    best_actions = argmax(q_values, dims=1)  # Actions selected by main network
    # Extract Q-values for the actions selected by main network
    return [target_q_values[best_actions[i]] for i in 1:batch_size]
end

"""
    update_target_network!(target_network, source_network)

Copy weights from the main Q-network to the target network.

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
function train_step!(q_network, target_network, optimizer, replay_buffer, batch_size, γ, env_actions, target_evaluation)
    # Sample a random batch of transitions from experience replay
    batch = sample(replay_buffer, batch_size)
    
    # Helper functions for converting the batch into matrix form for efficient computation
    states() = reduce(hcat, getfield.(batch, :state))  # (features, batch_size)
    action_indices() = [findfirst(==(action), env_actions) for action in getfield.(batch, :action)]
    next_states() = reduce(hcat, getfield.(batch, :next_state))  # (features, batch_size)
    rewards() = getfield.(batch, :reward)  # (batch_size,)
    terminal_values() = Float32.(getfield.(batch, :terminal))  # (batch_size,)
    
    # Compute target Q-values using the target network
    target_Q_values = target_network(next_states())  # (n_actions, batch_size)
    max_Q_values = target_evaluation(q_network, target_network, next_states(), target_Q_values, batch_size)
    
    # Bellman target: Q*(s,a) = r if next state is terminal or Q*(s,a) = r + γ * max_a' Q*(s',a') otherwise
    Q_target = rewards() + (1 .- terminal_values()) .* γ .* max_Q_values
        
    # Compute Q-values and loss using automatic differentiation
    loss, grads = Flux.withgradient(q_network) do model
        Q = model(states())  # (n_actions, batch_size)
        # Extract Q-values for the actions that were actually taken
        Q_prediction = [Q[action_indices()[i], i] for i in 1:batch_size]
        # Use Huber loss (less sensitive to outliers than MSE)
        Flux.Losses.huber_loss(Q_target, Q_prediction)
    end
    
    # Update the network weights using the computed gradients
    Flux.update!(optimizer, q_network, grads[1])
    
    return loss
end

"""
    DQN(env; kwargs...) -> Flux.Chain

Train an agent using the (Double) DQN algorithm.

# Arguments
- `env`: Environment implementing CommonRLInterface
- `hidden_layers=[128, 64]`: Architecture of hidden layers
- `η=1e-4`: Learning rate for Adam optimizer
- `γ=0.99`: Discount factor for Bellman equation
- `T=20_000`: Maximum steps per episode
- `ε=(0.5, 0.01)`: Epsilon tuple (initial, final) for exploration
- `Δε=1e-4`: Epsilon decay per episode
- `replay_memory_size=1_000_000`: Size of experience replay buffer
- `replay_start_size=100_000`: Start training after this many experiences
- `batch_size=32`: Batch size for training
- `update_frequency=4`: How often to perform training steps
- `target_evaluation=ddqn_target_evaluation`: Function for target Q-value evaluation
- `target_update_frequency=25_000`: How often to update target network
- `max_episodes=100_000`: Maximum number of episodes to train
- `callback=EpisodeLogger()`: Function called after each episode

# Returns
- Trained Q-network (Flux.Chain)

# Examples

## Basic usage (Double DQN):
```julia
env = BreakoutEnv()
q_network = DQN(env)
```

## Simple DQN:
```julia
env = BreakoutEnv()
q_network = DQN(env, target_evaluation=dqn_target_evaluation)
```
"""
function DQN(env; 
    hidden_layers=[128, 64], η=1e-4, 
    γ=0.99, T=20_000,  
    ε=(0.5, 0.01), Δε = 1e-4, 
    replay_memory_size=1_000_000, replay_start_size=100_000, batch_size=32, update_frequency=4,
    target_evaluation=ddqn_target_evaluation,  target_update_frequency=25_000,
    max_episodes=100_000,  
    callback=EpisodeLogger()
)
    RL.reset!(env)
    # Initialize experience replay buffer
    replay_buffer = ReplayBuffer(replay_memory_size)
    
    # Create the neural network architecture with flattened game state as input and action Q-values as output
    layers = [length(RL.observe(env)), hidden_layers..., length(RL.actions(env))]
    q_network = ValueNetwork(layers)        # Main Q-network
    target_network = ValueNetwork(layers)   # Target network for stable learning
    
    # Initialize target network with same weights as main network
    update_target_network!(target_network, q_network)
    
    # Set up Adam optimizer for gradient-based learning
    optimizer = Flux.setup(Adam(η), q_network)

    # Initialize update counters
    next_update = replay_start_size
    next_target_update = replay_start_size + target_update_frequency

    εᵢ = first(ε)  # Will be reduced by Δε after every episode

    watch_keypress() # allow to interrupt training by pressing ENTER key
    # Loop over episodes
    for i in 1:max_episodes
        # Reset environment for episode i
        RL.reset!(env)
        Sₜ₋₁ = RL.observe(env)
        ∑rₜ = 0.0
        
        t = 0
        # Loop over at most T steps within episode
        while t < T
            t += 1
            next_update -= 1
            next_target_update -= 1
            
            # εᵢ-greedy action selection
            if rand() < εᵢ
                Xₜ = rand(RL.actions(env)) # Random action
            else               
                Xₜ = RL.actions(env)[argmax(q_network(reshape(Sₜ₋₁, :, 1)))]  # Greedy action 
            end
            
            # Execute the chosen action in the environment
            rₜ = RL.act!(env, Xₜ)
            Sₜ = RL.observe(env)
            terminal = RL.terminated(env)
            
            # Store experience in replay buffer for later learning
            push!(replay_buffer, StateTransition(Sₜ₋₁, Xₜ, rₜ, Sₜ, terminal))
            
            # Perform training step when counter reaches zero
            if next_update <= 0
                loss = train_step!(q_network, target_network, optimizer, replay_buffer, batch_size, γ, RL.actions(env), target_evaluation)
                next_update = update_frequency  # Reset counter
            end
            
            # Update target network when counter reaches zero
            if next_target_update <= 0
                update_target_network!(target_network, q_network)
                next_target_update = target_update_frequency  # Reset counter
            end
            
            # Update episode reward
            ∑rₜ += rₜ
            Sₜ₋₁ = Sₜ
            
            # End episode if terminal state reached
            if terminal
                break
            end
        end
        
        # Callback for monitoring progress
        if callback !== nothing
            callback(i, t, ∑rₜ)
        end

        # Reduce εᵢ for reduced exploration and increased exploitation
        εᵢ = max(last(ε), εᵢ - Δε)

        if QUIT[]
            break
        end
    end
    
    return q_network
end
