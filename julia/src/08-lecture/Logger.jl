using DataStructures
using Plots

export EpisodeLogger

"""
    EpisodeLogger(window_size=1000; plot=true)

Callback logger for tracking and visualizing DQN training progress.

Maintains a moving average of episode rewards and optionally displays plots during training.
Always stores episode data for later plot generation using `create_plot!`.

# Arguments
- `window_size=1000`: Size of circular buffer for moving average calculation
- `plot=true`: Whether to display plots during training (every 10 episodes)

# Fields
- `reward_buffer`: Circular buffer for episode rewards (for moving average)
- `cumulative_reward_sum`: Running sum for efficient moving average calculation
- `episode_rewards`: All episode rewards for plotting
- `avg_rewards`: Moving average rewards for plotting
- `episodes`: Episode numbers for plotting
- `total_steps`: Cumulative step count across all episodes
- `plot`: Current plot object (updated during training or via `create_plot!`)
- `plot_enabled`: Whether to display plots during training

# Usage
```julia
# With live plotting
logger = EpisodeLogger(500, plot=true)
q_network = DQN(env, callback=logger)

# Without live plotting (generate plot after training)
logger = EpisodeLogger(500, plot=false)
q_network = DQN(env, callback=logger)
create_plot!(logger)  # Generate final plot
display(logger.plot)  # Show the plot
```
"""
mutable struct EpisodeLogger
    reward_buffer::CircularBuffer{Float32}
    cumulative_reward_sum::Float32
    episode_rewards::Vector{Float32}
    avg_rewards::Vector{Float32}
    episodes::Vector{Int}
    total_steps::Int
    plot::Union{Nothing, Plots.Plot}
    
    function EpisodeLogger(window_size::Int=1000; plot::Bool=true)
        reward_buffer = CircularBuffer{Float32}(window_size)
        empty_plot = plot ? Plots.plot() : nothing
        new(reward_buffer, 0.0f0, Float32[], Float32[], Int[], 0, empty_plot)
    end
end

function (logger::EpisodeLogger)(episode::Int, steps::Int, episode_reward::Float64)
    logger.total_steps += steps
    # Update moving average
    if length(logger.reward_buffer) == capacity(logger.reward_buffer)
        logger.cumulative_reward_sum -= first(logger.reward_buffer)
    end
    push!(logger.reward_buffer, episode_reward)
    logger.cumulative_reward_sum += episode_reward
    
    avg_reward = round(logger.cumulative_reward_sum / length(logger.reward_buffer), digits=2)
    println("Episode $episode, Total steps: $(logger.total_steps), Reward: $episode_reward, Avg($(length(logger.reward_buffer))): $avg_reward")
    
    push!(logger.episode_rewards, Float32(episode_reward))
    push!(logger.avg_rewards, Float32(avg_reward))
    push!(logger.episodes, episode)
    
    # Update plot every 10 episodes to avoid too frequent updates
    if logger.plot !== nothing && episode % 10 == 0 
        try
            create_plot!(logger)
            display(logger.plot)
        catch e
            @warn "Plotting failed: $e"
        end
    end
end

"""
    create_plot!(logger::EpisodeLogger)

Create and store a plot from the logger's stored data.

Updates `logger.plot` with a complete visualization showing episode rewards as scatter points
and moving average as a line. The plot includes proper legends and formatting.

# Usage
```julia
logger = EpisodeLogger(plot=false)  # No live plotting
q_network = DQN(env, callback=logger)
create_plot!(logger)  # Generate final plot
display(logger.plot)  # Show the plot
```
"""
function create_plot!(logger::EpisodeLogger)
    if isempty(logger.episodes)
        logger.plot = plot()  # Return empty plot if no data
        return
    end
    
    logger.plot = plot(logger.episodes, logger.episode_rewards, 
             label=false, color=:gray, seriestype=:scatter, 
             markersize=1, markerstrokewidth=0, markeralpha=1, 
             markerstrokecolor=:gray, xformatter=:plain)
    xlabel!(logger.plot, "Episode")
    ylabel!(logger.plot, "Reward")
    
    # Add moving average if we have enough data
    if length(logger.episodes) >= capacity(logger.reward_buffer)
        start_idx = findfirst(x -> x >= capacity(logger.reward_buffer), logger.episodes)
        if start_idx !== nothing
            episodes_subset = logger.episodes[start_idx:end]
            avg_rewards_subset = logger.avg_rewards[start_idx:end]
            plot!(logger.plot, episodes_subset, avg_rewards_subset, 
                  label="Average reward (last $(capacity(logger.reward_buffer)) episodes)", 
                  color=:blue, linewidth=1)
        end
    else
        # Add empty series for legend consistency
        plot!(logger.plot, Float64[], Float64[], 
              label="Average reward (last $(capacity(logger.reward_buffer)) episodes)", 
              color=:blue, linewidth=1)
    end
    
    # Add scatter points to legend
    plot!(logger.plot, [0], [0], label="Reward (episode)", color=:gray, 
          seriestype=:scatter, markersize=1, markerstrokewidth=0, 
          markerstrokecolor=:gray)
    
    plot!(logger.plot, legend=:topleft, legendfontsize=8, 
          legend_background_color=:white, legend_foreground_color=:black)
#    title!(logger.plot, "Training Progress")
end
