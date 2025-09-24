"""
Keyboard control module for Breakout.

Provides human player input through keyboard controls.
"""
module Control

using GameZero
using ..GameLogic

export get_action

"""
    get_action(g::Game) -> Int

Get paddle movement action from keyboard input.

# Returns
- `-1`: Move paddle left (LEFT arrow or A key)
- `1`: Move paddle right (RIGHT arrow or D key)  
- `0`: No movement
"""
function get_action(g::Game)
    if g.keyboard.LEFT || g.keyboard.A
        return -1
    elseif g.keyboard.RIGHT || g.keyboard.D
        return 1
    else
        return 0
    end
end

end # module Control