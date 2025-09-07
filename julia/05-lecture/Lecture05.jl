module Lecture05

# Export main functions for public use
export plot_stock_with_moving_average, demo_moving_average
export sliding_filter, apply_filter, demo_image_filtering, kernels
export create_lenet5
export demo_lenet5_mnist
export load_svhn_raw, load_svhn_data, demo_lenet5_svhn, check_svhn_download_progress 

# Include all the implementation files
include("sliding_filter.jl") 
include("image_filtering.jl")
include("moving_average.jl")
include("../04-lecture/mnist_data.jl")
include("LeNet5.jl")
include("svhn_data.jl")
include("Demo_MNIST.jl")
include("Demo_SVHN.jl") 

end
