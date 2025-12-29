using Flux
using Statistics
import CommonRLInterface as RL
using ..Lecture08: EpisodeLogger, QUIT, enable_interrupt


"""
    REINFORCE(env; kwargs...) -> Flux.Chain

Train an agent using the REINFORCE policy gradient algorithm.

# Arguments
- `env`: Environment implementing CommonRLInterface
- `hidden_layers=[64, 32]`: Architecture of hidden layers
- `η=1e-3`: Learning rate
- `γ=0.99`: Discount factor for returns
- `T=20_000`: Maximum steps per episode
- `max_episodes=1000`: Maximum number of episodes to train
- `callback=EpisodeLogger()`: Function called after each episode

# Returns
- Trained policy network (Flux.Chain)

# Example
```julia
env = BreakoutEnv()
policy = REINFORCE(env, max_episodes=1000)
```
"""
function REINFORCE(env; 
    hidden_layers=[64, 32], η=1e-3, 
    γ=0.99, T=20_000, max_episodes=100_000, 
    callback=EpisodeLogger()
)
    
    # Initialize environment and get dimensions
    RL.reset!(env)
    
    # Create policy network for discrete actions
    layers = [length(RL.observe(env)), hidden_layers..., length(RL.actions(env))]
    policy = Network(layers)
    
    # Set up optimizer (Adam works well for policy gradients)
    optimizer = Flux.setup(Adam(η), policy)
    
    enable_interrupt() # allow to interrupt training by pressing ENTER key
    # Training loop
    for i in 1:max_episodes
        # Reset environment for episode i
        RL.reset!(env)
        Sₜ₋₁ = RL.observe(env)
        ∑rₜ = 0.0
        
        trajectory = Tuple{Vector{Float32}, Int32, Float32}[] # (state,action,reward)

        t = 0
        # Loop over at most T steps within episode
        while t < T
            t += 1
            
            # Sample action from policy
            π = Flux.softmax(policy(reshape(Sₜ₋₁, :, 1))[:, 1])            
            action_idx = sample_action(π)

            Xₜ = RL.actions(env)[action_idx]
            rₜ = RL.act!(env, Xₜ)
            Sₜ = RL.observe(env)
            terminal = RL.terminated(env)          
            
            push!(trajectory, (Sₜ₋₁, Xₜ, rₜ))
            ∑rₜ += rₜ
            Sₜ₋₁ = Sₜ

            # End episode if terminal state reached
            if terminal
                break
            end
        end
        
        # Callback for logging
        if callback !== nothing
            callback(i, t, ∑rₜ)
        end

               ∇J̃ = Flux.gradient(policy) do model
            J̃ = 0.0f0 # Trajectory-based estimate of objective J(θ)
            R = 0.0f0
            while t > 0
                Sₜ₋₁, Xₜ, rₜ = trajectory[t] 
                R = rₜ + γ * R
                log_π = Flux.logsoftmax(model(reshape(Sₜ₋₁, :, 1))[:, 1])
                action_idx = findfirst(==(Xₜ), RL.actions(env))
                J̃ -= R * log_π[action_idx] # negative objective for gradient descent
                t -= 1
            end
            return J̃ / length(trajectory)  # Average objective per step
        end
              
        Flux.update!(optimizer, policy, ∇J̃[1])         

        if QUIT[]
            break
        end
    end
    
    return policy
end


