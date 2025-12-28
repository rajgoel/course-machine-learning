using Flux
using Statistics
import CommonRLInterface as RL
using ..Lecture08: EpisodeLogger, ValueNetwork, QUIT, enable_interrupt


"""
    ActorCritic(env; kwargs...) -> Flux.Chain

Train an agent using the Actor-Critic algorithm.

Actor-Critic combines policy gradients (actor) with value function learning (critic).
Updates are performed at each step using TD error as the advantage estimate.

# Arguments
- `env`: Environment implementing CommonRLInterface
- `hidden_layers=[64, 32]`: Architecture of hidden layers for both actor and critic
- `η=1e-4`: Learning rate for actor
- `η_critic=1e-3`: Learning rate for critic
- `γ=0.99`: Discount factor for TD error
- `T=20_000`: Maximum steps per episode
- `max_episodes=1000`: Maximum number of episodes to train
- `batch_size=32`: Number of steps to collect before updating networks
- `callback=EpisodeLogger()`: Function called after each episode

# Returns
- Trained policy network (Flux.Chain)

# Example
```julia
env = BreakoutEnv()
policy = ActorCritic(env, max_episodes=1000)
```
"""
function ActorCritic(env; 
    hidden_layers=[64, 32], η=1e-4, η_critic=1e-3,
    γ=0.99, T=20_000, max_episodes=100_000, 
    batch_size=32, callback=EpisodeLogger()
)
    
    # Initialize environment and get dimensions
    RL.reset!(env)
    feature_dim = length(RL.observe(env))
    n_actions = length(RL.actions(env))
    
    # Create actor (policy) and critic (value) networks
    actor_layers = [feature_dim, hidden_layers..., n_actions]
    critic_layers = [feature_dim, hidden_layers..., 1]
    
    actor = PolicyNetwork(actor_layers)    # π_θ(s,a)
    critic = ValueNetwork(critic_layers)   # V_θ_critic(s)
    
    # Set up optimizers
    actor_optimizer = Flux.setup(Adam(η), actor)
    critic_optimizer = Flux.setup(Adam(η_critic), critic)
    
    enable_interrupt() # allow to interrupt training by pressing ENTER key
    # Training loop
    for i in 1:max_episodes
        # Reset environment for episode i
        RL.reset!(env)
        Sₜ₋₁ = RL.observe(env)
        ∑rₜ = 0.0
        
        # Batch storage
        states = Vector{Vector{Float32}}()
        actions = Vector{Int}()
        advantages = Vector{Float32}()
        targets = Vector{Float32}()
        
        t = 0
        batch_count = 0
        
        # Loop over at most T steps within episode
        while t < T
            t += 1
            
            # Sample action from policy (actor)
            π = Flux.softmax(actor(reshape(Sₜ₋₁, :, 1))[:, 1])            
            action_idx = sample_action(π)
            Xₜ = RL.actions(env)[action_idx]
            
            # Execute action and observe next state
            rₜ = RL.act!(env, Xₜ)
            Sₜ = RL.observe(env)
            terminal = RL.terminated(env)
            
            # Compute TD error and targets
            V_curr = critic(reshape(Sₜ₋₁, :, 1))[1]
            V_next = terminal ? 0.0f0 : critic(reshape(Sₜ, :, 1))[1]
            target = rₜ + γ * V_next
            advantage = target - V_curr
            
            # Store in batch
            push!(states, copy(Sₜ₋₁))
            push!(actions, action_idx)
            push!(advantages, advantage)
            push!(targets, target)
            batch_count += 1
            
            # Update networks when batch is full or episode ends
            if batch_count >= batch_size || terminal
                # Batch update actor
                ∇_actor = Flux.gradient(actor) do model
                    loss = 0.0f0
                    for j in 1:batch_count
                        log_π = Flux.logsoftmax(model(reshape(states[j], :, 1))[:, 1])
                        loss += -advantages[j] * log_π[actions[j]]
                    end
                    loss / batch_count
                end
                
                # Batch update critic
                ∇_critic = Flux.gradient(critic) do model
                    loss = 0.0f0
                    for j in 1:batch_count
                        V_pred = model(reshape(states[j], :, 1))[1]
                        loss += (targets[j] - V_pred)^2
                    end
                    loss / batch_count
                end
                
                Flux.update!(actor_optimizer, actor, ∇_actor[1])
                Flux.update!(critic_optimizer, critic, ∇_critic[1])
                
                # Clear batch
                empty!(states)
                empty!(actions) 
                empty!(advantages)
                empty!(targets)
                batch_count = 0
            end
            
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

        if QUIT[]
            break
        end
    end
    
    return actor
end


