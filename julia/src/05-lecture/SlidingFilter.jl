"""
    sliding_filter(data, kernel; operation=sum, stride=1)

Generic sliding filter for N-dimensional data.

# Arguments
- `data::AbstractArray`: Input array (1D, 2D, 3D, etc.)
- `kernel::AbstractArray`: Filter kernel (same dimensionality as data)
- `operation::Function`: Function to apply (sum, mean, maximum, etc.) (default: sum)
- `stride::Int`: Step size for sliding window (default: 1)

# Returns
- `AbstractArray`: Filtered output array with reduced dimensions

# Example
```julia
# 1D sliding average
data = [1, 2, 3, 4, 5]
kernel = [0.5, 0.5]
result = sliding_filter(data, kernel; operation=sum)
```
"""
function sliding_filter(data, kernel; operation=sum, stride=1)
    
    data_size = size(data)
    kernel_size = size(kernel)
    
    # Calculate output dimensions
    output_size = tuple([div(data_size[i] - kernel_size[i], stride) + 1 
                        for i in 1:ndims(data)]...)
    
    # Initialize output
    output = similar(data, output_size)
    
    # Generate all possible starting positions
    ranges = [1:stride:data_size[i]-kernel_size[i]+1 for i in 1:ndims(data)]
    positions = Iterators.product(ranges...)
    
    # Apply filter at each position
    for (idx, pos) in enumerate(positions)
        # Extract window
        window_ranges = [pos[i]:pos[i]+kernel_size[i]-1 for i in 1:ndims(data)]
        window = data[window_ranges...]
        
        # Apply operation
        output[idx] = operation(window .* kernel)
    end
    
    return output
end
