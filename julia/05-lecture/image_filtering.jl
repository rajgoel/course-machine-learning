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
Apply image filtering using sliding filters
Color images are automatically converted to grayscale before filtering
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
    filtered = sliding_filter(image_matrix, kernel, sum)
    
    # Normalize to [0,1] range for display
    filtered = abs.(filtered)
    filtered = filtered ./ maximum(filtered)
    
    return filtered
end

"""
Demonstrate image filtering with a given image and kernel
Color images are automatically converted to grayscale before filtering
"""
function demo_image_filtering(image_path, kernel=kernels[:laplacian])
    # Load image
    if !isfile(image_path)
        error("Image file '$image_path' not found! Please check the file path.")
    end
    image = load(image_path)
    
    # Apply the given kernel
    filtered = apply_filter(image, kernel)

    # Display and return the processed image
    result = Gray.(filtered)
    
    # Try to display if GUI is available (skip in CI environments)
    try
        # Check if ImageView is available and use it
        ImageView = Base.require(Main, :ImageView)
        ImageView.imshow(result)
    catch e
        println("Display not available (headless environment). Image processing completed.")
    end
    
    return result
end

# Usage examples:
# demo_image_filtering("my_image.png")
# demo_image_filtering("my_image.png", kernels[:emboss])
# demo_image_filtering("my_image.png", [0 -1 0; -1 5 -1; 0 -1 0])

