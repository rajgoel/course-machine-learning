"""
Imitation Learning for Breakout using FluxDNN

Implements supervised learning approach where a FluxDNN learns to mimic
expert (heuristic) behavior by mapping game states to actions.
"""

# Use modules included in Lecture08
using MachineLearningCourse.Lecture04
using Flux
using Statistics
using Random
using JLD2

export train_imitation, run_imitation

"""
    ImitationAgent

Wrapper for trained FluxDNN that implements the Control interface.
"""
mutable struct ImitationAgent
    network::FluxDNN
    action_classes::Vector{Int}  # [-1, 0, 1]
    
    function ImitationAgent(network::FluxDNN)
        new(network, [-1, 0, 1])
    end
end

"""
    get_action(agent::ImitationAgent, game_state) -> Int

Get action from trained imitation learning agent.

# Arguments
- `agent::ImitationAgent`: Trained imitation learning agent
- `game_state`: Current game state tuple

# Returns
- `Int`: Action (-1, 0, 1) for paddle movement
"""
function get_action(agent::ImitationAgent, game_state)
    # Flatten state to neural network input format
    state_vector = Breakout.flatten(game_state)
    
    # Get network prediction (softmax probabilities)
    action_probs = predict(agent.network, reshape(state_vector, :, 1))
    
    # Convert to action index (1-based)
    action_idx = argmax(action_probs[:, 1])
    
    # Map to actual action (-1, 0, 1)
    return agent.action_classes[action_idx]
end

"""
    collect_expert_data(num_episodes::Int=100, max_steps_per_episode::Int=1000) 
                       -> (Vector{Vector{Float32}}, Vector{Vector{Float32}})

Collect expert demonstration data using heuristic agent.

Runs the heuristic expert for multiple episodes and collects state-action pairs.
States are flattened game representations, actions are one-hot encoded.

# Arguments
- `num_episodes::Int`: Number of episodes to collect (default: 100)
- `max_steps_per_episode::Int`: Maximum steps per episode (default: 1000)

# Returns
- `Tuple`: (states, actions)
  - `states`: Vector of flattened state vectors (each ROWS*COLS + 5 elements)
  - `actions`: Vector of one-hot encoded actions (3 elements each)
"""
function collect_expert_data(num_episodes::Int=100, max_steps_per_episode::Int=1000)
    println("Collecting expert demonstration data...")
    println("Episodes: $num_episodes, Max steps per episode: $max_steps_per_episode")
    
    # Use heuristic expert from Breakout module
    expert_action = Breakout.get_heuristic_action
    
    # Storage for collected data
    states = Vector{Float32}[]
    actions = Vector{Float32}[]
    
    # Enable simulation mode (no auto-restart)
    
    total_steps = 0
    episodes_completed = 0
    
    for episode in 1:num_episodes
        Breakout.reset(false)  # Reset game state
        episode_steps = 0
        
        while episode_steps < max_steps_per_episode
            # Get current game state
            game_state = Breakout.get_game_state()
            
            # Get expert action from heuristic
            action = expert_action(game_state)
            
            # Store state-action pair
            state_vector = Breakout.flatten(game_state)
            action_vector = action_to_onehot(action)
            
            push!(states, state_vector)
            push!(actions, action_vector)
            
            # Update game with action
            game_over = !Breakout.update(action)
            
            episode_steps += 1
            total_steps += 1
            
            # Break if game over
            if game_over
                break
            end
        end
        
        episodes_completed += 1
        if episode % 10 == 0
            println("  Episode $episode/$num_episodes completed, total steps: $total_steps")
        end
    end
    
    # Disable simulation mode
    
    println("Data collection completed!")
    println("  Episodes: $episodes_completed")
    println("  Total state-action pairs: $(length(states))")
    println("  Average steps per episode: $(round(total_steps / episodes_completed, digits=1))")
    
    return states, actions
end

"""
    action_to_onehot(action::Int) -> Vector{Float32}

Convert action integer to one-hot encoded vector.

# Arguments
- `action::Int`: Action value (-1, 0, or 1)

# Returns
- `Vector{Float32}`: One-hot encoded action [left, stay, right]
"""
function action_to_onehot(action::Int)
    if action == -1
        return Float32[1, 0, 0]  # Move left
    elseif action == 0
        return Float32[0, 1, 0]  # Stay
    elseif action == 1
        return Float32[0, 0, 1]  # Move right
    else
        error("Invalid action: $action. Must be -1, 0, or 1.")
    end
end

"""
    train_imitation_agent(states, actions; hidden_layers=[128, 64], learning_rate=0.001, 
                         epochs=100, batch_size=128, validation_split=0.2, verbose=true)
                         -> ImitationAgent

Train FluxDNN to imitate expert behavior via supervised learning.

# Arguments
- `states`: Vector of state vectors from expert demonstrations  
- `actions`: Vector of one-hot action vectors from expert demonstrations
- `hidden_layers`: Hidden layer dimensions (default: [128, 64])
- `learning_rate`: Adam optimizer learning rate (default: 0.001)
- `epochs`: Number of training epochs (default: 100)
- `batch_size`: Mini-batch size (default: 128)
- `validation_split`: Fraction of data for validation (default: 0.2)
- `verbose`: Print training progress (default: true)

# Returns
- `ImitationAgent`: Trained agent ready for evaluation
"""
function train_imitation_agent(states, actions; 
                              hidden_layers=[128, 64], 
                              learning_rate=0.001,
                              epochs=100, 
                              batch_size=128,
                              validation_split=0.2,
                              verbose=true)
    
    println("Training imitation learning agent...")
    println("Architecture: $(length(states[1])) → $(join(hidden_layers, " → ")) → 3")
    println("Training samples: $(length(states))")
    
    # Convert data to matrices
    input_dim = length(states[1])
    output_dim = 3  # Three actions: left, stay, right
    
    X = hcat(states...)  # (features × samples)
    Y = hcat(actions...) # (actions × samples)
    
    println("Data shapes: X=$(size(X)), Y=$(size(Y))")
    
    # Create train/validation split
    n_samples = size(X, 2)
    n_train = round(Int, n_samples * (1 - validation_split))
    
    # Shuffle indices
    indices = randperm(n_samples)
    train_idx = indices[1:n_train]
    val_idx = indices[n_train+1:end]
    
    X_train = X[:, train_idx]
    Y_train = Y[:, train_idx]
    X_val = X[:, val_idx]
    Y_val = Y[:, val_idx]
    
    println("Training set: $(size(X_train, 2)) samples")
    println("Validation set: $(size(X_val, 2)) samples")
    
    # Create network architecture
    layers = [input_dim, hidden_layers..., output_dim]
    network = FluxDNN(layers)
    
    println("Total parameters: $(sum(length, Flux.trainables(network.model)))")
    
    # Train the network
    println("\nStarting training...")
    losses = train!(network, X_train, Y_train, learning_rate, epochs; 
                   batch_size=batch_size, verbose=verbose)
    
    # Evaluate on validation set
    if length(X_val) > 0
        println("\nEvaluating on validation set...")
        val_accuracy = accuracy(network, X_val, Y_val)
        println("Validation accuracy: $(round(val_accuracy*100, digits=2))%")
    end
    
    # Create and return trained agent
    agent = ImitationAgent(network)
    println("\nImitation learning agent training completed!")
    
    return agent
end

"""
    evaluate_agent(agent::ImitationAgent, num_episodes::Int=10) -> Dict

Evaluate trained agent performance in Breakout environment.

# Arguments
- `agent::ImitationAgent`: Trained agent to evaluate
- `num_episodes::Int`: Number of evaluation episodes (default: 10)

# Returns
- `Dict`: Evaluation metrics (Breakout.scores, episode lengths, etc.)
"""
function evaluate_agent(agent::ImitationAgent, num_episodes::Int=10)
    println("Evaluating trained agent...")
    println("Episodes: $num_episodes")
    
    # Enable simulation mode
    
    Breakout.scores = Int[]
    episode_lengths = Int[]
    
    for episode in 1:num_episodes
        Breakout.reset(false)
        episode_steps = 0
        initial_Breakout.score = Breakout.score
        
        while episode_steps < 2000  # Max steps per episode
            # Get current state and agent action
            game_state = Breakout.get_game_state()
            action = get_action(agent, game_state)
            
            # Update game with action
            game_over = !Breakout.update(action)
            
            episode_steps += 1
            
            # Break if game over
            if game_over
                break
            end
        end
        
        final_Breakout.score = Breakout.score
        episode_Breakout.score = final_Breakout.score - initial_Breakout.score
        
        push!(Breakout.scores, episode_Breakout.score)
        push!(episode_lengths, episode_steps)
        
        if episode % 2 == 0 || episode == num_episodes
            println("  Episode $episode: Score = $episode_Breakout.score, Length = $episode_steps")
        end
    end
    
    # Disable simulation mode
    
    # Compute statistics
    results = Dict(
        "mean_Breakout.score" => mean(Breakout.scores),
        "std_Breakout.score" => std(Breakout.scores),
        "max_Breakout.score" => maximum(Breakout.scores),
        "min_Breakout.score" => minimum(Breakout.scores),
        "mean_length" => mean(episode_lengths),
        "std_length" => std(episode_lengths),
        "Breakout.scores" => Breakout.scores,
        "lengths" => episode_lengths
    )
    
    println("\nEvaluation Results:")
    println("  Mean Score: $(round(results["mean_Breakout.score"], digits=1)) ± $(round(results["std_Breakout.score"], digits=1))")
    println("  Score Range: $(results["min_Breakout.score"]) - $(results["max_Breakout.score"])")
    println("  Mean Episode Length: $(round(results["mean_length"], digits=1)) steps")
    
    return results
end

"""
    save_agent(agent::ImitationAgent, filepath::String)

Save trained agent to disk using JLD2.

# Arguments
- `agent::ImitationAgent`: Trained agent to save
- `filepath::String`: Path to save the agent (e.g., "imitation_agent.jld2")
"""
function save_agent(agent::ImitationAgent, filepath::String)
    println("Saving agent to: $filepath")
    
    # Create directory if it doesn't exist
    mkpath(dirname(filepath))
    
    # Save agent data
    jldsave(filepath; 
        network_layers = agent.network.layers,
        network_state = Flux.state(agent.network.model),
        action_classes = agent.action_classes
    )
    
    println("Agent saved successfully!")
end

"""
    load_agent(filepath::String) -> ImitationAgent

Load trained agent from disk using JLD2.

# Arguments
- `filepath::String`: Path to the saved agent file

# Returns
- `ImitationAgent`: Loaded agent ready for use
"""
function load_agent(filepath::String)
    println("Loading agent from: $filepath")
    
    # Load agent data
    data = load(filepath)
    
    # Reconstruct network
    network = FluxDNN(data["network_layers"])
    Flux.loadmodel!(network.model, data["network_state"])
    
    # Create agent
    agent = ImitationAgent(network)
    agent.action_classes = data["action_classes"]
    
    println("Agent loaded successfully!")
    return agent
end

"""
    train_imitation(; kwargs...)

Convenient interface function for training imitation learning agent.
"""
function train_imitation(; num_episodes=50, max_steps_per_episode=500,
                        hidden_layers=[32], learning_rate=0.001,
                        epochs=50, batch_size=64, validation_split=0.2,
                        model_path="imitation_agent.jld2", verbose=true)
    
    println("="^80)
    println("TRAINING IMITATION LEARNING AGENT")
    println("="^80)
    
    # Collect expert demonstrations
    println("\nCollecting expert demonstrations...")
    states, actions = collect_expert_data(num_episodes, max_steps_per_episode)
    
    # Train the agent
    println("\nTraining neural network...")
    agent = train_imitation_agent(
        states, actions;
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose
    )
    
    # Save the model
    println("\nSaving trained model...")
    full_path = isabspath(model_path) ? model_path : joinpath(@__DIR__, model_path)
    save_agent(agent, full_path)
    
    println("\nTraining completed! ✅")
    return agent
end

"""
    run_imitation(; model_path="imitation_agent.jld2")

Convenient interface function for running Breakout with imitation agent.
"""
function run_imitation(; model_path="imitation_agent.jld2")
    println("="^60)
    println("BREAKOUT WITH IMITATION LEARNING AGENT")
    println("="^60)
    
    # Check if model exists
    full_path = isabspath(model_path) ? model_path : joinpath(@__DIR__, model_path)
    
    if !isfile(full_path)
        println("❌ Model not found: $full_path")
        println("Please train a model first using train_imitation()")
        return
    end
    
    println("Loading trained agent and starting game...")
    println("Press ESC to quit")
    
    # Load the trained agent
    agent = load_agent(full_path)
    
    # Create control function for the agent
    function agent_control(game_state)
        return get_action(agent, game_state)
    end
    
    # Use the Breakout module that's already loaded
    println("Using imitation agent control...")
    Breakout.breakout(agent_control)  # Default autorestart=true for interactive play
end