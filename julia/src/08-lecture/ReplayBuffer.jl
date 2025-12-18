using DataStructures

"""
    StateTransition

Represents a single experience tuple (s, a, r, s', done) for the replay buffer.

This is the fundamental unit of experience that the DQN agent learns from.
Each transition contains:
- `state`: The current state observation (flattened game state)
- `action`: The action taken by the agent
- `reward`: The immediate reward received
- `next_state`: The resulting state after taking the action
- `terminal`: Whether the episode ended (true if game over)
"""
struct StateTransition
    state::Vector{Float32}
    action::Int32
    reward::Float32
    next_state::Vector{Float32}
    terminal::Bool
end

const ReplayBuffer = CircularBuffer{StateTransition}

"""
    sample(replay_buffer::ReplayBuffer, batch_size::Int) -> Vector{StateTransition}

Randomly sample a batch of experiences from the replay buffer.

This implements the experience replay mechanism from the DQN paper.
By randomly sampling past experiences, we break the temporal correlations
that would otherwise make learning unstable.

# Arguments
- `replay_buffer`: Circular buffer containing past experiences
- `batch_size`: Number of experiences to sample

# Returns
- Vector of `StateTransition` objects for training
"""
function sample(replay_buffer::ReplayBuffer, batch_size::Int)
    # Randomly sample indices without replacement
    indices = rand(1:length(replay_buffer), batch_size)
    return replay_buffer[indices]
end

