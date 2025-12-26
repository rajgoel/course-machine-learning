using Breakout
import CommonRLInterface as RL

"""
    demo(; algorithm=ActorCritic, max_episodes=1000, plot=true)

Run policy gradient training demo on Breakout environment.

# Arguments
- `algorithm=ActorCritic`: Algorithm function to use (ActorCritic or REINFORCE)
- `max_episodes=500_000`: Maximum number of training episodes
- `plot=true`: Whether to display live plots during training

# Returns
- Function-dependent return values

# Usage
```julia
# Actor-Critic (default)
result = demo(max_episodes=500)

# REINFORCE
result = demo(algorithm=REINFORCE, max_episodes=500)
```
"""
function demo(algorithm=ActorCritic; max_episodes=100_000, plot=true)
    env = BreakoutEnv(:minimal)
    logger = EpisodeLogger(plot=plot)

    println("Policy Gradient Training initialized:")
    println("  Algorithm: ", algorithm)
    println("  Actions: ", RL.actions(env))
    println("  Max episodes: ", max_episodes)
    println("  Live plotting: ", plot)

    result = algorithm(env, max_episodes=max_episodes, callback=logger)
    return result, logger
end

