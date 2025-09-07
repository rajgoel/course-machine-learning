"""
LeNet-5 CNN Architecture Implementation using Flux.jl

This implementation follows the original LeNet-5 architecture (but uses ReLU instead of sigmoid):
- Conv2D (6 filters, 5x5) + ReLU + AvgPool (2x2)
- Conv2D (16 filters, 5x5) + ReLU + AvgPool (2x2)
- Flatten
- Dense (120) + ReLU
- Dense (84) + ReLU
- Dense (10) + Softmax
"""

using Flux

"""
    create_lenet5(height, width, channels)

Create the LeNet-5 CNN architecture using Flux.jl. The original LeNet-5 architecture uses sigmoid, here ReLU is used.

# Parameters
- `height`: Input height in pixels
- `width`: Input width in pixels
- `channels`: The number of input channels - 1 for grayscale, 3 for RGB

# Returns
- Tuple: (model, layer_dimensions)
  - model: Flux.Chain LeNet-5 model  
  - layer_dimensions: Array of tuples showing tensor dimensions at each layer

The architecture consists of:
1. Convolutional layer: 6 filters of size 5×5, ReLU activation
2. Average pooling: 2×2 pool size
3. Convolutional layer: 16 filters of size 5×5, ReLU activation
4. Average pooling: 2×2 pool size
5. Flatten layer
6. Dense layer: calculated size → 120 units, ReLU activation
7. Dense layer: 120 → 84 units, ReLU activation
8. Output layer: 84 → 10 units (for 10 digit classes)
"""
function create_lenet5(height, width, channels)
    # Calculate layer dimensions
    layer_dimensions = Vector{Any}(undef, 9)  # Pre-allocate for 9 layers
    
    # Layer 1: Input layer
    layer_dimensions[1] = (height=height, width=width, channels=channels)
 
    # Layer 2: Convolution (5x5, 6 filters)
    layer_dimensions[2] = (height=layer_dimensions[1].height - 4, width=layer_dimensions[1].width - 4, channels=6)

    # Layer 3: Average pooling (2x2)
    layer_dimensions[3] = (height=div(layer_dimensions[2].height, 2), width=div(layer_dimensions[2].width, 2), channels=6)
    
    # Layer 4: Convolution (5x5, 16 filters)
    layer_dimensions[4] = (height=layer_dimensions[3].height - 4, width=layer_dimensions[3].width - 4, channels=16)

    # Layer 5: Average pooling (2x2)
    layer_dimensions[5] = (height=div(layer_dimensions[4].height, 2), width=div(layer_dimensions[4].width, 2), channels=16)
    
    # Layer 6: Flattened
    layer_dimensions[6] = (size=layer_dimensions[5].height * layer_dimensions[5].width * layer_dimensions[5].channels,)

    # Layer 7: Dense (120)
    layer_dimensions[7] = (size=120,)

    # Layer 8: Dense (84)
    layer_dimensions[8] = (size=84,)

    # Layer 9: Dense (10)
    layer_dimensions[9] = (size=10,)                                          
    
    model = Flux.Chain(
        # First convolutional block
        Flux.Conv((5, 5), channels => 6, Flux.relu),
        Flux.MeanPool((2, 2)),
        
        # Second convolutional block  
        Flux.Conv((5, 5), 6 => 16, Flux.relu),
        Flux.MeanPool((2, 2)),
        
        # Flatten and fully connected layers
        Flux.flatten,
        Flux.Dense(layer_dimensions[6].size, layer_dimensions[7].size, Flux.relu),
        Flux.Dense(layer_dimensions[7].size, layer_dimensions[8].size, Flux.relu),
        Flux.Dense(layer_dimensions[8].size, layer_dimensions[9].size),
        Flux.softmax
    )
    
    return model, layer_dimensions
end

