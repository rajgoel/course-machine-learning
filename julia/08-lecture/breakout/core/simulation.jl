module Simulation

# Import only collision detection from GameZero, no graphics
using GameZero: collide, Rect
using Statistics: mean
using ..GameLogic: reset, set_simulation_mode, get_game_state, handle_input, update_physics, 
                   score, ball_x, ball_y, ball_vel, paddle, bricks, BALL_SIZE, GAME_HEIGHT
using ..Control: get_action
using ..Screenshot: screenshot

export batch_simulate

function simulate(;
    max_steps=10000,
    save_states=false,
    state_interval=100,
    save_pixels=false,
    pixel_interval=1000
)
    """Run game simulation without display
    
    Args:
        max_steps: Maximum simulation steps
        save_states: Whether to save game states periodically
        state_interval: Steps between state saves
        save_pixels: Whether to save pixel representations
        pixel_interval: Steps between pixel saves
    
    Returns:
        Dict with simulation results
    """
    
    # Enable simulation mode (no auto-restart on game over)
    set_simulation_mode(true)
    reset()
    
    # Initialize storage if requested
    states = save_states ? [] : nothing
    pixel_states = save_pixels ? [] : nothing
    initial_score = score
    
    current_step = 0
    for i in 1:max_steps
        current_step = i
        # Get action from agent
        action = get_action(nothing)
        
        # Update game
        handle_input(action)
        game_continues = update_physics()
        
        # Save states if requested
        if save_states && current_step % state_interval == 0
            push!(states, get_game_state())
        end
        
        # Save pixel states if requested
        if save_pixels && current_step % pixel_interval == 0
            game_state = get_game_state()
            pixels = screenshot(game_state)
            push!(pixel_states, pixels)
        end
        
        # Check for game over
        if !game_continues
            break  # Game over (ball fell off bottom)
        end
    end
    
    return Dict(
        :final_score => score,
        :score_gained => score - initial_score,
        :final_state => get_game_state(),
        :steps => current_step,
        :bricks_remaining => length(bricks),
        :game_over => current_step < max_steps,
        :level_complete => false,  # Never ends on level complete, continues to next level
        :saved_states => states,
        :saved_pixels => pixel_states
    )
end

function batch_simulate(episodes=100; kwargs...)
    """Run multiple simulations and collect statistics
    
    Returns:
        Dict with aggregated results
    """
    results = []
    
    for episode in 1:episodes
        result = simulate(; kwargs...)
        push!(results, result)
        
        if episode % 10 == 0
            println("Completed $episode/$episodes simulations")
        end
    end
    
    # Calculate statistics  
    scores = [r[:final_score] for r in results]
    steps = [r[:steps] for r in results]
    completions = sum([r[:level_complete] for r in results])
    
    return Dict(
        :episodes => episodes,
        :results => results,
        :mean_score => mean(scores),
        :max_score => maximum(scores),
        :min_score => minimum(scores),
        :mean_steps => mean(steps),
        :completion_rate => completions / episodes,
        :total_completions => completions
    )
end

end # module Simulation
