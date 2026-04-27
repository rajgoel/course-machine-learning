"""
Sampling utilities for reinforcement learning.
"""

"""
    sample_action(probs::Vector)

Sample action from probability distribution using cumulative distribution.

# Arguments
- `probs`: Action probabilities (should sum to 1.0)

# Returns
- Action index (Int) sampled according to probabilities

# Example
```julia
probs = [0.1, 0.7, 0.2]  # Action probabilities
action_idx = sample_action(probs)  # Returns 1, 2, or 3
```
"""
function sample_action(probs)
    cumsum_probs = cumsum(probs)
    r = rand()
    action_idx = findfirst(p -> p >= r, cumsum_probs)
    if action_idx === nothing  # Numerical precision fix
        action_idx = length(probs)
    end
    return action_idx
end