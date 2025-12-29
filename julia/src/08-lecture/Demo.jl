using Breakout
import CommonRLInterface as RL

"""
    demo(algorithm=:DDQN; max_episodes=100_000, plot=true)

Run a DQN training demo on Breakout environment.

# Arguments
- `algorithm=:DDQN`: Selected algorithm (use `:DQN` for standard DQN)
- `max_episodes=100_000`: Maximum number of training episodes
- `plot=true`: Whether to display live plots during training

# Returns
- `(q_network, logger)`: Trained network and logger

# Usage
```julia
# With live plotting
q_network, logger = Lecture08.demo(max_episodes=1000, plot=true)

# DQN without live plotting (faster training)
q_network, logger = Lecture08.demo(:DQN, max_episodes=1000, plot=false)
Lecture08.create_plot!(logger)  # Generate final plot
using Plots
plot!(logger.plot,size=(800,600)) # Resize plot
savefig(logger.plot,"training.png")
```
"""
function demo(algorithm=:DDQN; max_episodes=100_000, plot=true)
    env = BreakoutEnv(:brickless)
    logger = EpisodeLogger(plot=plot)

    println("  Algorithm: ", algorithm)
    println("  Max episodes: ", max_episodes)

    if algorithm == :DDQN
        target_evaluation=ddqn_target_evaluation
    elseif algorithm == :DQN
        target_evaluation=dqn_target_evaluation
    else 
        error("Unknown algorithm: $algorithm. Supported algorithms are :DQN and :DDQN")
    end

    q_network = DQN(env, target_evaluation=target_evaluation, max_episodes=max_episodes, callback=logger)
    return q_network, logger
end

"""
    breakout(model_path::String; speed=nothing)

Run Breakout game with a trained DQN model.

# Arguments
- `model_path`: Path to saved DQN model (.jld2 file)
- `speed=nothing`: Game speed

# Example
```julia
# Run game with trained model
breakout("agent.jld2")
```
"""
function breakout(model_path::String=joinpath(@__DIR__, "agent.jld2"); speed=nothing)
    # Load the trained model
    model = load(model_path)
    
    # Run Breakout with the DQN agent
    Breakout.breakout(game_state -> agent(game_state, model), speed=speed)
end
