using Breakout
import CommonRLInterface as RL

"""
    demo(; max_episodes=100_000, target_evaluation=ddqn_target_evaluation, plot=true)

Run a DQN training demo on Breakout environment.

# Arguments
- `max_episodes=100_000`: Maximum number of training episodes
- `target_evaluation=ddqn_target_evaluation`: Target evaluation function (use `dqn_target_evaluation` for standard DQN)
- `plot=true`: Whether to display live plots during training

# Returns
- `(q_network, logger)`: Trained network and logger with training data

# Usage
```julia
# With live plotting
q_network, logger = demo(max_episodes=1000, plot=true)

# Without live plotting (faster training)
q_network, logger = demo(max_episodes=1000, plot=false)
create_plot!(logger)  # Generate final plot
display(logger.plot)  # Show the plot
```
"""
function demo(; max_episodes=100_000, target_evaluation=ddqn_target_evaluation, plot=true)
    env = BreakoutEnv()
    logger = EpisodeLogger(plot=plot)

    # Print training configuration for educational purposes
    println("Environment initialized:")
    println("  Actions: ", RL.actions(env))
    println("  Max episodes: ", max_episodes)
    println("  Live plotting: ", plot)

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
breakout("trained_breakout_dqn.jld2")
```
"""
function breakout(model_path::String=joinpath(@__DIR__, "models", "ddqn_agent.jld2"); speed=nothing)
    # Load the trained model
    q_network = load(model_path)
    
    # Run Breakout with the DQN agent
    Breakout.breakout(game_state -> dqn_agent(game_state, q_network), speed=speed)
end
