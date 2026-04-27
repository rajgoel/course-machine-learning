using Random
using Statistics
using LinearAlgebra
using REPL.TerminalMenus

"""
    movie_explorer(embeddings, data)

Interactive menu for exploring movie similarities using trained embeddings.

# Arguments
- `embeddings::Matrix{Float32}`: Trained node embeddings from GCN
- `data`: Dataset containing graph and metadata

# Example
```julia
_, embeddings, data = Lecture06.demo(interactive=false)
Lecture06.movie_explorer(embeddings, data)
```
"""
function movie_explorer(embeddings, data)
    println("\n" * "="^60)
    println("Movie explorer")
    println("="^60)
    
    # Get all movie names for the menu, sorted alphabetically
    movie_start = data.n_users + 1
    movie_indices = collect(movie_start:length(data.train_graph.nodes))
    movie_data = [(i, data.train_graph.nodes[i].name) for i in movie_indices]
    sort!(movie_data, by=x -> x[2])  # Sort by movie name
    movie_indices = [x[1] for x in movie_data]
    movie_names = [x[2] for x in movie_data]
    
    while true
        println("\nSelect a movie to find similar ones:")
        
        # Create menu with movie names
        menu_options = [movie_names; "Quit"]
        menu = RadioMenu(menu_options, pagesize=20)
        choice = request(menu)
        
        if choice == length(menu_options) || choice == -1  # Quit or cancelled
            println("Goodbye!")
            break
        end
        
        # Get selected movie
        movie_idx = movie_indices[choice]
        target_movie_name = movie_names[choice]
        
        # Find and display similar movies
        println("\n" * "="^60)
        println("MOVIES SIMILAR TO: $target_movie_name")
        println("="^60)
        
        # Get all other movie indices (excluding target)
        other_movie_indices = [i for i in movie_indices if i != movie_idx]
        target_indices = fill(movie_idx, length(other_movie_indices))
        
        # Compute similarities directly
        similarity_scores = similarities(embeddings[target_indices, :], embeddings[other_movie_indices, :])
        
        # Create and sort results with movie names
        results = []
        for i in 1:length(other_movie_indices)
            idx = other_movie_indices[i]
            score = similarity_scores[i]
            movie_name = data.train_graph.nodes[idx].name
            push!(results, (idx, movie_name, score))
        end
        sort!(results, by=x -> x[3], rev=true)
        
        println("\n🎬 TOP 10 SIMILAR MOVIES:")
        for i in 1:min(10, length(results))
            idx, name, score = results[i]
            println("  $i. $(round(score, digits=3)) similarity - $name")
        end
        
        println("\nPress Enter to continue...")
        readline()
    end
end

"""
    demo(dataset_size="100k"; epochs=100, embedding_sizes=[128, 64], interactive=false)

Demonstrate collaborative filtering using GCN on MovieLens data.

# Arguments
- `dataset_size`: MovieLens dataset size ("100k" or "1m") (default: "100k")
- `epochs`: Number of training epochs (default: 100)
- `embedding_sizes::Vector{Int}`: Embedding dimensions for each layer (default: [28, 64])
- `η::Float64`: Learning rate for Adam optimizer (default: 0.001)
- `batch_size::Int`: Mini-batch size for SGD (default: 1024)
- `interactive::Bool`: If true, prompt for movie index to find similar movies (default: true)

# Returns
- `Tuple`: (model, embeddings, data, losses, test_rmse)

# Example
```julia
# Run with default 100k dataset
Lecture06.demo()
```

```julia
# Learn embeddings without exploring similar movies
model, embeddings, data = Lecture06.demo(interactive=false)
```

```julia
# Run with 1m dataset and custom architecture (slow)
Lecture06.demo("1m", embedding_sizes=[256, 128, 64])
```
"""
function demo(dataset_size="100k"; epochs=100, embedding_sizes=[128, 64], η=0.001, batch_size=1024, interactive=true)
    println("MovieLens GCN Demo")
    println("="^60)
    
    # Load MovieLens data
    data = load_movielens_data(dataset_size)
    
    # Use GCN with Graph struct
    model, embeddings, losses = GCN(
        data.train_graph;
        epochs=epochs,
        embedding_sizes=embedding_sizes,
        η=η,
        batch_size=batch_size
    )
    
    # Compute test RMSE using generic prediction
    test_origin_indices = [edge.origin_index for edge in data.test_graph.edges]
    test_destination_indices = [edge.destination_index for edge in data.test_graph.edges]
    test_predictions = similarities(embeddings[test_origin_indices, :], embeddings[test_destination_indices, :])
    test_actual = [edge.weight for edge in data.test_graph.edges]
    test_rmse = sqrt(mean((test_predictions .- test_actual).^2))
    
    println("Final test error (RMSE): $(round(test_rmse, digits=4)) ratings")
    
    # Interactive mode for exploring similar movies
    if interactive
        movie_explorer(embeddings, data)
    end
    
    return model, embeddings, data
end
