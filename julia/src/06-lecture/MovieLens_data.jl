using MLDatasets
using Random
using Statistics
using DataFrames

"""
    load_movielens_data(dataset_size="100k"; train_ratio=0.8)

Load MovieLens dataset using MLDatasets.jl and split into train/test sets.

# Arguments
- `dataset_size::String`: Dataset size ("100k", "1m", "10m", "20m") (default: "100k")
- `train_ratio::Float32`: Proportion for training set (default: 0.8f0)

# Returns
- NamedTuple with train_graph, test_graph, n_users, n_items
"""
function load_movielens_data(dataset_size="100k"; train_ratio=0.8)
    println("Loading MovieLens $dataset_size dataset...")
    
    # Load data using MLDatasets
    dataset = MovieLens(dataset_size)
    
    # Extract ratings data from standard MovieLens format
    ratings = dataset.graphs[1]
    rating_edge_key = ("user", "rating", "movie")
    
    # Get user-movie connections and rating values
    user_ids, item_ids = ratings.edge_indices[rating_edge_key]
    raw_rating_values = ratings.edge_data[rating_edge_key][:rating]
    
    # Center and normalize ratings: (1-5) -> (-1,+1)
    rating_values = (Float32.(raw_rating_values) .- 3.0f0) ./ 2.0f0
    
    # Get dataset statistics
    n_users = maximum(user_ids)
    n_items = maximum(item_ids)
    n_ratings = length(rating_values)
    
    println("Dataset statistics:")
    println("  Users: $n_users")
    println("  Items: $n_items") 
    println("  Ratings: $n_ratings")
    println("  Rating range: $(extrema(rating_values))")
    println("  Average rating: $(round(mean(rating_values), digits=2))")
    println("  Sparsity: $(round(100 * (1 - n_ratings / (n_users * n_items)), digits=2))%")
    
    # Split into train/test
    Random.seed!(42)
    n_train = Int(round(n_ratings * train_ratio))
    
    shuffled_indices = randperm(n_ratings)
    train_indices = shuffled_indices[1:n_train]
    test_indices = shuffled_indices[n_train+1:end]
    
    println("Data split:")
    println("  Train: $n_train ratings")
    println("  Test: $(n_ratings - n_train) ratings")
    
    # Create mapping from original IDs to consecutive indices
    unique_user_ids = unique(user_ids)
    unique_item_ids = unique(item_ids)
    
    # Create node-to-index mappings
    user_to_idx = Dict(uid => i for (i, uid) in enumerate(unique_user_ids))
    item_to_idx = Dict(iid => i + length(unique_user_ids) for (i, iid) in enumerate(unique_item_ids))
    
    # Get movie titles from dataset metadata
    movie_titles = dataset.metadata["movie_id_to_title"]
    
    # Create Node objects with consecutive indices
    user_nodes = [Node(i, "User $(unique_user_ids[i])") for i in 1:length(unique_user_ids)]
    movie_nodes = [Node(i + length(unique_user_ids), movie_titles[unique_item_ids[i]]) 
                   for i in 1:length(unique_item_ids)]
    
    all_nodes = [user_nodes; movie_nodes]
    
    # Convert to Edge structs using consecutive indices (all edges are bidirectional)  
    train_edges = [Edge(user_to_idx[user_ids[i]], item_to_idx[item_ids[i]], rating_values[i]) for i in train_indices]
    test_edges = [Edge(user_to_idx[user_ids[i]], item_to_idx[item_ids[i]], rating_values[i]) for i in test_indices]
    
    # Create Graph structs
    train_graph = Graph(all_nodes, train_edges)
    test_graph = Graph(all_nodes, test_edges)  # Same nodes, different edges
    
    println("Created Graph structures:")
    println("  Nodes: $(length(all_nodes))")
    println("  Train edges: $(length(train_edges))")
    println("  Test edges: $(length(test_edges))")
    
    return (
        train_graph = train_graph,
        test_graph = test_graph,
        n_users = length(unique_user_ids),  # Number of unique users in graph
        n_items = length(unique_item_ids),   # Number of unique items in graph
        dataset_info = dataset  # Additional dataset metadata
    )
end

