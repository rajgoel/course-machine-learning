"""
Breakout Simulation Runner

Provides simulation interface for running Breakout episodes without graphics.
Ideal for reinforcement learning experiments, data collection, and automated testing.
"""

using FileIO, Images, Colors, Statistics
import Pkg

"""
    simulate(control_file="control/heuristic.jl"; episodes=1, max_steps=10000, save_pixels=false, pixel_interval=1000, save_states=false, state_interval=500)

Run Breakout simulation episodes with specified control and recording options.

# Arguments
- `control_file`: AI control module to use
- `episodes`: Number of episodes to run (1 = single episode with detailed output, >1 = batch with summary)
- `max_steps`: Maximum steps per episode
- `save_pixels`: Whether to save pixel state images
- `pixel_interval`: Steps between pixel saves
- `save_states`: Whether to save game state snapshots  
- `state_interval`: Steps between state saves

# Returns
- Single episode: Episode result dictionary with detailed metrics
- Multiple episodes: Batch summary with aggregated statistics
"""
function simulate(control_file="control/heuristic.jl";
                 episodes=1,
                 max_steps=10000, 
                 save_pixels=false, 
                 pixel_interval=1000, 
                 save_states=false, 
                 state_interval=500)
    
    println("🎮 Running Breakout simulation with control: $control_file")
    
    # Switch to breakout environment temporarily  
    original_env = Base.active_project()
    breakout_dir = @__DIR__
    
    try
        # Activate breakout environment
        Pkg.activate(breakout_dir)
        if !isfile(joinpath(breakout_dir, "Manifest.toml"))
            println("Installing breakout dependencies...")
            Pkg.instantiate()
        end
        
        # Include all required modules (use absolute paths)
        include(joinpath(breakout_dir, "core/digits.jl"))
        include(joinpath(breakout_dir, "core/game_logic.jl")) 
        include(joinpath(breakout_dir, "core/screenshot.jl"))
        include(joinpath(breakout_dir, control_file))
        include(joinpath(breakout_dir, "core/simulation.jl"))
        
        # Import the modules at function level - use eval to make it work
        Core.eval(@__MODULE__, :(using .GameLogic, .Control, .Screenshot, .Simulation))
        
        # Run simulation using Base.invokelatest to handle world age issue
        batch_result = Base.invokelatest(Simulation.batch_simulate, episodes,
            max_steps=max_steps,
            save_pixels=save_pixels,
            pixel_interval=pixel_interval,
            save_states=save_states,
            state_interval=state_interval
        )
        
        # For single episode, return just the episode result and print details
        if episodes == 1
            episode_result = batch_result[:results][1]
            println("\n📊 Simulation Results:")
            println("Final Score: $(episode_result[:final_score])")
            println("Steps Taken: $(episode_result[:steps])")
            println("Bricks Remaining: $(episode_result[:bricks_remaining])")
            println("Game Over: $(episode_result[:game_over])")
            println("Level Complete: $(episode_result[:level_complete])")

            # Save pixel images if requested
            if save_pixels && episode_result[:saved_pixels] !== nothing
                println("\n💾 Saving pixel images...")
                for (i, pixels) in enumerate(episode_result[:saved_pixels])
                    step_number = i * pixel_interval
                    filename = "simulation_step_$(step_number).png"
                    save(filename, Gray.(pixels))
                    println("  Saved: $filename")
                end
            end

            # Save final state
            if save_pixels
                println("\n🎯 Saving final game state...")
                final_pixels = screenshot(episode_result[:final_state])
                save("simulation_final.png", Gray.(final_pixels))
                println("  Saved: simulation_final.png")
            end

            println("\n✅ Simulation complete!")
            return episode_result
        else
            # For multiple episodes, return the batch summary
            return batch_result
        end
        
    finally
        # Switch back to original environment
        Pkg.activate(original_env)
    end
end
