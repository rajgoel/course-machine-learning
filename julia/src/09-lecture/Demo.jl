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
- `(policy, logger)`: Trained network and logger

# Usage
```julia
# REINFORCE (default)
policy, logger = Lecture09.demo(max_episodes=500)

# Actor-critic
policy, logger = Lecture09.demo(algorithm=ActorCritic, max_episodes=500)
```
"""
function demo(algorithm=:REINFORCE; max_episodes=100_000, plot=true)
    env = BreakoutEnv(:brickless)
    logger = EpisodeLogger(plot=plot)

    println("  Algorithm: ", algorithm)
    println("  Max episodes: ", max_episodes)

    if algorithm == :REINFORCE
        policy = REINFORCE(env, max_episodes=max_episodes, callback=logger)
    elseif algorithm == :ActorCritic
        policy = ActorCritic(env, max_episodes=max_episodes, callback=logger)
    else 
        error("Unknown algorithm: $algorithm. Supported algorithms are :REINFORCE and :ActorCritic")
    end

    return policy, logger
end

