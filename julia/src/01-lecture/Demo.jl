"""
    demo(a=nothing)

Simple demo for 2x2 pixel symbol recognition

# Arguments
- `a`: input vector of 4 real numbers. If not provided, random values between 0 and 1 are used.

# Usage
```julia
# Run with random input
demo()

# Run with specific 2x2 pixel intensities arranged as [top-left, top-right, bottom-left, bottom-right]
demo([0.8, 0.2, 0.1, 0.9])
```

The demo implements a simple linear classifier that takes an input representing pixel intensities to identify either symbol ⟋ or ⟍ .
"""
function demo(a=nothing)
    println("Simple Neural Network Demo")
    println("="^40)
    
    if a === nothing
        a = rand(4)
    elseif !(isa(a, Vector{<:Real}) && length(a) == 4)
        error("Input 'a' must be a vector of 4 real numbers")
    end
    
    # Input
    # Weights and biases
    W = [0.5  0.5  0.5 -0.5;
         0.5  0.5 -0.5  0.5]
    b = [0, 0]

    # Output    
    â = W * a + b
    
    println("Input:")
    println("┌─────┬─────┐")
    println("│ $(round(a[1], digits=1)) │ $(round(a[2], digits=1)) │")
    println("├─────┼─────┤")
    println("│ $(round(a[3], digits=1)) │ $(round(a[4], digits=1)) │")
    println("└─────┴─────┘")
    println("Output:")
    if â[1] >= â[2]
        println("┌───┐")
        println("│ ⟋ │")
        println("└───┘")
    end
    if â[1] ≈ â[2]
        println("or")
    end
    if â[2] >= â[1]
        println("┌───┐")
        println("│ ⟍ │")
        println("└───┘")
    end
    return â
end

