using SimpleDirectMediaLayer
using SimpleDirectMediaLayer.LibSDL2

# Window constants - game dimensions scaled by 4
const SCALE_FACTOR = 4
const WINDOW_WIDTH = GAME_WIDTH * SCALE_FACTOR
const WINDOW_HEIGHT = GAME_HEIGHT * SCALE_FACTOR

# Module-level variables
window = nothing
renderer = nothing

function create_window()
    global window, renderer
    @assert SDL_Init(SDL_INIT_EVERYTHING) == 0 "error initializing SDL: $(unsafe_string(SDL_GetError()))"
    
    window = SDL_CreateWindow("Breakout", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE)
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED)
    
    # Set logical size for automatic scaling and centering
    SDL_RenderSetLogicalSize(renderer, GAME_WIDTH, GAME_HEIGHT)
    
    println("Window created.")
end

function close_window()
    global window, renderer
    if renderer !== nothing
        SDL_DestroyRenderer(renderer)
        renderer = nothing
    end
    if window !== nothing
        SDL_DestroyWindow(window)
        window = nothing
    end
    SDL_Quit()
    println("Window closed.")
end

function process_events(game_state=nothing)
    event_ref = Ref{SDL_Event}()
    while Bool(SDL_PollEvent(event_ref))
        evt = event_ref[]
        if evt.type == SDL_QUIT
            return false  # Signal to stop
        elseif evt.type == SDL_KEYDOWN
            if evt.key.keysym.scancode == SDL_SCANCODE_ESCAPE
                return false  # ESC key also stops
            elseif evt.key.keysym.scancode == SDL_SCANCODE_P
                # Take screenshot when P is pressed
                if game_state !== nothing
                    try
                        save_screenshot_png(game_state)
                    catch e
                        println("Error saving screenshot: $e")
                    end
                end
            end
        end
    end
    return true  # Continue
end

function render_display(game_state, current_game)
    # 1. Clear the screen
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255)  # Black background
    SDL_RenderClear(renderer)
    
    # 2. Draw game objects using shared drawing function
    if game_state !== nothing
        draw_game(game_state, (x, y, color) -> begin
            # Convert to SDL drawing call
            r, g, b, a = color
            SDL_SetRenderDrawColor(renderer, r, g, b, a)
            pixel_rect = SDL_Rect(x-1, y-1, 1, 1)  # Convert 1-based to 0-based coordinates
            SDL_RenderFillRect(renderer, Ref(pixel_rect))
        end, current_game)
    end
    
    # 3. Present the frame
    SDL_RenderPresent(renderer)
end

"""
    render_screenshot(game_state) -> Array{Float64, 2}

Render game state as grayscale pixel array for headless gameplay and RL training.

This function creates a visual representation of the game state without requiring 
SDL rendering, making it suitable for reinforcement learning agents and headless 
training environments.

# Arguments
- `game_state`: Tuple of (score, ball_cx, ball_cy, ball_vx, ball_vy, paddle_cx, bricks)

# Returns
- 2D array (GAME_HEIGHT × GAME_WIDTH) with grayscale values 0.0-1.0
"""
function render_screenshot(game_state, current_game=1)
    # Initialize empty game field
    pixels = zeros(Float64, GAME_HEIGHT, GAME_WIDTH)
    
    # Use shared drawing function with pixel array callback
    draw_game(game_state, (x, y, color) -> begin
        if x >= 1 && x <= GAME_WIDTH && y >= 1 && y <= GAME_HEIGHT
            # Convert RGBA color to grayscale using luminance formula
            r, g, b, a = color
            gray_value = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            pixels[y, x] = gray_value
        end
    end, current_game)
    
    return pixels
end

"""
    save_screenshot_png(game_state, filename="breakout_screenshot.png")

Save a screenshot of the current game state as a PNG file.

# Arguments
- `game_state`: Current game state tuple
- `filename`: Output filename (default: "breakout_screenshot.png")
"""
function save_screenshot_png(game_state, filename="breakout_screenshot.png")
    # Get grayscale pixel array
    pixels = render_screenshot(game_state)
    
    # Convert to 8-bit grayscale (0-255)
    img_array = round.(UInt8, pixels * 255)
    
    # Save as PNG using Images.jl
    save(filename, img_array)
    
    println("Screenshot saved: $filename")
end

