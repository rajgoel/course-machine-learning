"""
SVHN Dataset Loading and Preprocessing

This module provides SVHN (Street View House Numbers) dataset loading functionality
for RGB image classification using convolutional neural networks.
"""

using MLDatasets
using Random
using Flux
using Statistics

"""
    load_svhn_raw(train_size=5000, test_size=1000)

Load raw SVHN (Street View House Numbers) dataset without preprocessing.
Perfect for data exploration and visualization.

# Arguments
- `train_size::Int`: Number of training samples to use (default: 5000)  
- `test_size::Int`: Number of test samples to use (default: 1000)

# Returns
- `Tuple`: (X_train, Y_train, X_test, Y_test)
  - `X_train::Array{UInt8, 4}`: Training images (32 × 32 × 3 × samples), values 0-255
  - `Y_train::Vector{Int}`: Training labels as integers 1-10 (SVHN format: 1-9 for digits 1-9, 10 for digit 0)
  - `X_test::Array{UInt8, 4}`: Test images (32 × 32 × 3 × samples), values 0-255  
  - `Y_test::Vector{Int}`: Test labels as integers 1-10

# Example Usage - Image Visualization
```julia
# Load raw data
X_train, Y_train, X_test, Y_test = load_svhn_raw(100, 50)

# Display RGB images
using ImageView
show_image = i -> imshow(reverse(permutedims(reverse(X_train[:, :, :, i] ./ 255, dims=1), (2, 1, 3)), dims=2))
show_label = i -> Y_train[i] == 10 ? 0 : Y_train[i]  # Convert 10→0, keep 1-9
show_image(1)  # Show first image
show_label(1)  # Show first label
```
"""
function load_svhn_raw(train_size::Int=5000, test_size::Int=1000)
    println("Loading raw SVHN dataset ...")
    println()
    
    # Check if already downloaded
    svhn_found = false
    svhn_path = joinpath(first(Base.DEPOT_PATH), "scratchspaces")
    
    if isdir(svhn_path)
        for dir in readdir(svhn_path)
            full_dir = joinpath(svhn_path, dir, "datadeps", "SVHN2")
            if isdir(full_dir)
                required_files = ["train_32x32.mat", "test_32x32.mat", "extra_32x32.mat"]
                min_sizes = [180_000_000, 60_000_000, 1_300_000_000]
                
                all_files_present = true
                for (file, min_size) in zip(required_files, min_sizes)
                    filepath = joinpath(full_dir, file)
                    if !isfile(filepath) || filesize(filepath) < min_size
                        all_files_present = false
                        break
                    end
                end
                
                if all_files_present
                    svhn_found = true
                    println("✓ SVHN data found in cache...")
                    break
                end
            end
        end
    end
    
    if !svhn_found
        println("⬇ Downloading SVHN dataset for the first time...")
        println("  This includes 73,257 training + 26,032 test + 531,131 extra images")
        println("  Please be patient - Download is ~1.4GB and may take several minutes.")
        println("  Subsequent runs will be fast as data is cached locally.")
    end
    
    # Load raw SVHN data (no preprocessing)
    train_x, train_y = MLDatasets.SVHN2(split=:train)[:]
    test_x, test_y = MLDatasets.SVHN2(split=:test)[:]
    
    println("Original data shapes:")
    println("  Training: $(size(train_x)) images, $(length(train_y)) labels")
    println("  Test: $(size(test_x)) images, $(length(test_y)) labels")
    
    # Take subset but keep original format
    train_indices = randperm(size(train_x, 4))[1:train_size]
    test_indices = randperm(size(test_x, 4))[1:test_size]
    
    X_train = train_x[:, :, :, train_indices]  # Keep UInt8, 0-255
    Y_train = train_y[train_indices]           # Keep original 1-10 labels
    X_test = test_x[:, :, :, test_indices]     # Keep UInt8, 0-255
    Y_test = test_y[test_indices]              # Keep original 1-10 labels
    
    println("Raw data loaded:")
    println("  Training: $(size(X_train)) RGB images (32×32×3×$(train_size)), values 0-255")
    println("  Test: $(size(X_test)) RGB images (32×32×3×$(test_size)), values 0-255")
    println("  Labels: Integer format (1-9 for digits 1-9, 10 for digit 0)")
    
    return X_train, Y_train, X_test, Y_test
end

"""
    load_svhn_data(train_size=5000, test_size=1000)

Load and preprocess SVHN (Street View House Numbers) dataset.

# Arguments
- `train_size::Int`: Number of training samples to use (default: 5000)  
- `test_size::Int`: Number of test samples to use (default: 1000)

# Returns
- `Tuple`: (X_train, Y_train, X_test, Y_test)
  - `X_train::Array{Float32, 4}`: Training images (32 × 32 × 3 × samples)
  - `Y_train`: Training labels (10 × samples, one-hot encoded)
  - `X_test::Array{Float32, 4}`: Test images (32 × 32 × 3 × samples)
  - `Y_test`: Test labels (10 × samples, one-hot encoded)

SVHN contains 32×32 RGB images of street view house numbers (digits 0-9).
Images are normalized to [0,1] and labels are one-hot encoded for 10-class classification.
Returns 4D tensor format ready for CNN input: (height, width, channels, batch).

"""
function load_svhn_data(train_size::Int=5000, test_size::Int=1000)
    println("Loading SVHN dataset for CNN training...")
    
    # Load raw data first
    X_train_raw, Y_train_raw, X_test_raw, Y_test_raw = load_svhn_raw(train_size, test_size)
    
    println("Preprocessing for CNN training...")
    
    # Convert to Float32 and normalize to [0,1]
    X_train = Float32.(X_train_raw) ./ 255.0f0
    X_test = Float32.(X_test_raw) ./ 255.0f0
    
    # Convert labels from 1-10 to 0-9 (SVHN uses 1-10 for digits 1-9,0)
    Y_train_labels = Y_train_raw .- 1  # Convert 1-10 to 0-9
    Y_test_labels = Y_test_raw .- 1    # Convert 1-10 to 0-9
    
    # One-hot encode labels (Flux format: classes × samples)
    Y_train = Flux.onehotbatch(Y_train_labels, 0:9)
    Y_test = Flux.onehotbatch(Y_test_labels, 0:9)
    
    println("Preprocessed data:")
    println("  Training: $(size(X_train)) RGB images (32×32×3×$(train_size))")
    println("  Test: $(size(X_test)) RGB images (32×32×3×$(test_size))")
    println("  Labels: $(size(Y_train)) one-hot encoded (10×samples)")
    println("  Pixel values normalized to [0,1]")
    
    return X_train, Y_train, X_test, Y_test
end
