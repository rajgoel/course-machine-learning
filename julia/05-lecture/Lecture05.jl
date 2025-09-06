module Lecture05

# Export main functions for public use
export create_lenet5, train_lenet5, demo_lenet5
export sliding_filter, apply_filter, demo_image_filtering, kernels
export plot_stock_with_moving_average, demo_moving_average, load_companies

# Include all the implementation files
include("LeNet5.jl")
include("sliding_filter.jl") 
include("image_filtering.jl")
include("moving_average.jl")

end