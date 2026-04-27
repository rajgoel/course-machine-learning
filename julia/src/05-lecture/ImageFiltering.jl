using Images

# Available filter kernels for user access
const kernels = Dict(
    # Edge detection
    :sobel_x =>      [-1  0  1;
                      -2  0  2;
                      -1  0  1],

    :sobel_y =>      [-1 -2 -1;
                       0  0  0;
                       1  2  1], 

    :laplacian =>    [ 0 -1  0;
                      -1  4 -1;
                       0 -1  0],

    :edge_enhance => [-1 -1 -1;
                      -1  8 -1;
                      -1 -1 -1],

    # Blur filters
    :box_blur =>     [1 1 1;
                      1 1 1;
                      1 1 1] / 9,

    :gaussian_blur =>  [1 2 1;
                        2 4 2;
                        1 2 1] / 16,

    # Sharpen
    :sharpen =>      [ 0 -1  0;
                      -1  5 -1;
                       0 -1  0],

    # Emboss
    :emboss =>       [-2 -1  0;
                      -1  1  1;
                       0  1  2],
)

"""
    apply_filter(image, kernel)

Apply image filtering using sliding filters.

# Arguments
- `image`: Input image (color or grayscale)
- `kernel::AbstractMatrix`: Filter kernel to apply

# Returns
- `Gray`: Filtered grayscale image normalized to [0,1]

# Notes
Color images are automatically converted to grayscale before filtering.

# Example
```julia
using Images
img = load("image.jpg")
filtered = apply_filter(img, kernels[:edge_enhance])
```
"""
function apply_filter(image, kernel)
    # Resolve kernel
    if !isa(kernel, AbstractMatrix)
        error("Invalid filter")
    end
    
    # Convert image to grayscale if needed
    if eltype(image) <: AbstractRGB
        gray_image = Gray.(image)
        image_matrix = Float64.(gray_image)
    elseif ndims(image) == 3
        gray_image = Gray.(image)
        image_matrix = Float64.(gray_image)
    else
        image_matrix = Float64.(image)
    end
    
    # Apply edge detection filter
    filtered = sliding_filter(image_matrix, kernel; operation=sum)
    
    # Normalize to [0,1] range for display
    filtered = abs.(filtered)
    filtered = filtered ./ maximum(filtered)
    
    return Gray.(filtered)
end

"""
    filter_image(image_path=joinpath(@__DIR__, "containers_CC0.jpg"), kernel=kernels[:laplacian])

Demonstrate image filtering with a given image and kernel.

# Arguments  
- `image_path::String`: Path to image file (default: containers_CC0.jpg in current directory)
- `kernel::AbstractMatrix`: Filter kernel to apply (default: Laplacian edge detection)

# Returns
- `Gray`: Filtered grayscale image

# Notes
Color images are automatically converted to grayscale before filtering.

# Example
```julia
# Use default image and Laplacian filter
filtered = filter_image()

# Use custom image and Sobel filter
filtered = filter_image("my_image.png", kernels[:sobel_x])

# Display with ImageView
using ImageView
imshow(filtered)
```
"""
function filter_image(image_path=joinpath(@__DIR__, "containers_CC0.jpg"), kernel=kernels[:laplacian])
    # Load image
    if !isfile(image_path)
        error("Image file '$image_path' not found! Please check the file path.")
    end

    return apply_filter(load(image_path), kernel)
end

