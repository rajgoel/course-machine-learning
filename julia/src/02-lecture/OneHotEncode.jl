"""
    one_hot_encode(label, num_classes)

Convert class labels to one-hot vectors for classification.

# Arguments
- `label::Int`: Class label (1-indexed)
- `num_classes::Int`: Total number of classes

# Returns
- `Vector{Float64}`: One-hot encoded vector

# Example
```julia
one_hot_encode(3, 10)  # Returns [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
```
"""
function one_hot_encode(label::Int, num_classes::Int)
    encoded = zeros(Float64, num_classes)
    encoded[label] = 1.0
    return encoded
end

