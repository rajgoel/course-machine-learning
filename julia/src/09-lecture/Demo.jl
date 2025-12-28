using Breakout
import CommonRLInterface as RL

"""
    demo(algorithm=:REINFORCE; max_episodes=100_000, plot=true)

Run policy gradient training demo on Breakout environment.

# Arguments
- `algorithm=:REINFORCE`: Algorithm function to use (:REINFORCE or :ActorCritic)
- `max_episodes=100_000`: Maximum number of training episodes
- `plot=true`: Whether to display live plots during training

# Returns
- Function-dependent return values

# Usage
```julia
# REINFORCE (default)
result = Lecture09.demo(max_episodes=500)

# Actor-critic
result = Lecture09.demo(algorithm=ActorCritic, max_episodes=500)
```
"""
function demo(algorithm=:REINFORCE; max_episodes=100_000, plot=true)
    env = BreakoutEnv(:brickless)
    logger = EpisodeLogger(plot=plot)

    println("Policy Gradient Training initialized:")
    println("  Algorithm: ", algorithm)
    println("  Actions: ", RL.actions(env))
    println("  Max episodes: ", max_episodes)
    println("  Live plotting: ", plot)

    if algorithm == :REINFORCE
        result = REINFORCE(env, max_episodes=max_episodes, callback=logger)
    elseif algorithm == :ActorCritic
        result = ActorCritic(env, max_episodes=max_episodes, callback=logger)
    else 
        error("Unknown algorithm: $algorithm. Supported algorithms are :REINFORCE and :ActorCritic")
    end

    return result, logger
end

