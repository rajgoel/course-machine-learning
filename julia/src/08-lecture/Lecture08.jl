"""
    Lecture08

Implementation of Deep Q-Learning for the Breakout game

# Available Functions

- `demo()`: Train agent for the Breakout game using DQN
- `breakout()`: Play Train Breakout game using trained agent

# Usage

```julia
using MachineLearningCourse
q_network = Lecture08.demo()
Lecture08.save(q_network,"model.jld2")
Lecture08.breakout("model.jld2")
```

```julia
using MachineLearningCourse
Lecture08.breakout()
```
"""

module Lecture08

# Include components
using Breakout
include("Logger.jl")
include("ValueNetwork.jl")
include("ReplayBuffer.jl")
include("DQN.jl")
include("DQNAgent.jl")
include("Demo.jl")

export demo, breakout, create_plot!, DQN, dqn_target_evaluation, ddqn_target_evaluation, dqn_agent, save, load

end # module Lecture08
