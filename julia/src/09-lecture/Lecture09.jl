"""
    Lecture09

Policy Gradient Methods: REINFORCE and Actor-Critic

# Available Functions

- `demo_reinforce()`: Train agent using REINFORCE algorithm
- `reinforce_action()`: Get action from trained REINFORCE policy

# Usage

```julia
using MachineLearningCourse
policy, logger = Lecture09.demo_reinforce(max_episodes=500)
```
"""

module Lecture09

# Include components
using Breakout
using ..Lecture08: EpisodeLogger, create_plot!  # Import from Lecture08
include("PolicyNetwork.jl")
include("REINFORCE.jl")
include("ActorCritic.jl")
include("A2C.jl")
include("PolicyAgent.jl")
include("Demo.jl")

export demo, policy_agent, save, load, EpisodeLogger, create_plot!

end # module Lecture09
