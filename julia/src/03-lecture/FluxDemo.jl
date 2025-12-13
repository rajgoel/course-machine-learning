using Flux

"""
    flux_demo(;train_size=1000, epochs=10)

Minimal MNIST demonstration using gradient descent.

# Arguments  
- `train_size`: Number of training samples (default: 5000)
- `test_size`: Number of test samples (default: 1000)
- `epochs`: Number of training epochs (default: 1000)

# Returns
- Trained model and final accuracy
"""
function flux_demo(; train_size::Int=5000, test_size::Int=1000, epochs::Int=100)
    println("Minimal Flux.jl MNIST Demo")
    println("=" ^ 30)
    
    # Load data
    X_train_raw, Y_train_raw, X_test_raw, Y_test_raw = load_mnist_data(train_size, test_size)
    
    # Convert to Flux format (features × samples) and Float32
    X_train = reshape(X_train_raw, 784, train_size)  # 784 × train_size
    Y_train = Y_train_raw     # 10 × train_size  
    X_test = reshape(X_test_raw, 784, test_size)    # 784 × test_size
    Y_test = Y_test_raw       # 10 × test_size
    
    # Create simple model
    model = Flux.Chain(
        Flux.Dense(784 => 128, Flux.relu),
        Flux.Dense(128 => 64, Flux.relu),
        Flux.Dense(64 => 10)
    )
    
    # Setup training
    opt_state = Flux.setup(Flux.Adam(0.001), model)
    
    println("Training...")
    
    # Training loop using Flux.train!
    for epoch in 1:epochs
        Flux.train!(model, [(X_train, Y_train)], opt_state) do m, x, y
            Flux.logitcrossentropy(m(x), y)
        end
        
        train_loss = Flux.logitcrossentropy(model(X_train), Y_train)
        println("Epoch $epoch: Loss = $(round(train_loss, digits=4))")
    end
    
    # Evaluate
    predictions = Flux.softmax(model(X_test))  # Apply softmax for predictions
    pred_classes = [argmax(predictions[:, i]) - 1 for i in 1:size(predictions, 2)]
    true_classes = [argmax(Y_test[:, i]) - 1 for i in 1:size(Y_test, 2)]
    
    accuracy = sum(pred_classes .== true_classes) / length(pred_classes)
    println("Test Accuracy: $(round(accuracy * 100, digits=1))%")
    
    return model, accuracy
end
