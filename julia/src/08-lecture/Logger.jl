using DataStructures
using Plots

export EpisodeLogger

# Callback logger for episode completion
mutable struct EpisodeLogger
    reward_buffer::CircularBuffer{Float32}
    cumulative_reward_sum::Float32
    plot_enabled::Bool
    episode_rewards::Vector{Float32}
    avg_rewards::Vector{Float32}
    episodes::Vector{Int}
    epsilons::Vector{Float32}
    
    function EpisodeLogger(window_size::Int=1000; plot::Bool=true)
        reward_buffer = CircularBuffer{Float32}(window_size)
        new(reward_buffer, 0.0f0, plot, Float32[], Float32[], Int[], Float32[])
    end
end

function (logger::EpisodeLogger)(episode::Int, total_steps::Int, episode_reward::Float64, epsilon::Float64)
    # Update moving average
    if length(logger.reward_buffer) == capacity(logger.reward_buffer)
        logger.cumulative_reward_sum -= first(logger.reward_buffer)
    end
    push!(logger.reward_buffer, episode_reward)
    logger.cumulative_reward_sum += episode_reward
    
    avg_reward = round(logger.cumulative_reward_sum / length(logger.reward_buffer), digits=2)
    println("Episode $episode, Total steps: $total_steps, Reward: $episode_reward, Avg($(length(logger.reward_buffer))): $avg_reward")
    
    # Update plot if enabled
    if logger.plot_enabled
        push!(logger.episode_rewards, Float32(episode_reward))
        push!(logger.avg_rewards, Float32(avg_reward))
        push!(logger.episodes, episode)
        push!(logger.epsilons, Float32(epsilon))
        
        # Update plot every 10 episodes to avoid too frequent updates
        if episode % 10 == 0
            try
                p = plot(logger.episodes, logger.episode_rewards, label=false, color=:gray, seriestype=:scatter, markersize=1, markerstrokewidth=0, markeralpha=1, markerstrokecolor=:gray, xformatter=:plain)
                xlabel!(p, "Episode")
                ylabel!(p, "Reward")
                
                # Use plain number formatting
                plot!(p, xformatter=:plain)
                # Only plot moving average after buffer capacity episodes
                if episode >= capacity(logger.reward_buffer)
                    # Get only the episodes and avg rewards from episode capacity onwards
                    start_idx = findfirst(x -> x >= capacity(logger.reward_buffer), logger.episodes)
                    if start_idx !== nothing
                        episodes_subset = logger.episodes[start_idx:end]
                        avg_rewards_subset = logger.avg_rewards[start_idx:end]
                        plot!(p, episodes_subset, avg_rewards_subset, label="Average reward (last $(capacity(logger.reward_buffer)) episodes)", color=:blue, linewidth=1)
                    end
                else
                    # Add empty series for legend when no average is plotted yet
                    plot!(p, Float64[], Float64[], label="Average reward (last $(capacity(logger.reward_buffer)) episodes)", color=:blue, linewidth=1)
                end
                
                # Add reward entries to main legend manually
                plot!(p, [0], [0], label="Reward (episode)", color=:gray, seriestype=:scatter, markersize=1, markerstrokewidth=0, markerstrokecolor=:gray)
                
                # Add top margin and position legend above chart
#                plot!(p, top_margin=22Plots.PlotMeasures.mm)
                plot!(p, legend=:outertop, legendfontsize=8, 
                     legend_background_color=:white, legend_foreground_color=:black)
                
                title!(p, "Training Progress")
                display(p)
            catch e
                @warn "Plotting failed: $e"
            end
        end
    end
end
