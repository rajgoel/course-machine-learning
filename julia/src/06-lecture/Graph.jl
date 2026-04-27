using Flux
using Statistics
using LinearAlgebra
using SparseArrays

"""
    Node

Represents a node in a graph with an identifier and name.
"""
struct Node
    index::Int
    name::String
end

"""
    Edge

Represents a weighted edge in an undirected graph.
"""
struct Edge
    origin_index::Int
    destination_index::Int
    weight::Float32
end

"""
    Graph

Represents an undirected graph with nodes and edges.
"""
struct Graph
    nodes::Vector{Node}
    edges::Vector{Edge}
end

"""
    create_adjacency_matrix(graph::Graph; use_weights=false)

Creates the adjacency matrix of the graph.

# Arguments
- `graph::Graph`: Complete graph with nodes and edges using consecutive indices
- `use_weights::Bool`: If true, use edge weights; if false, use 1.0 for all edges (default: false)

# Returns
- `A`: Adjacency matrix
"""
function create_adjacency_matrix(graph::Graph; use_weights=false)
    origin_indices = Int[]
    destination_indices = Int[]
    weights = Float32[]
    
    for edge in graph.edges
        edge_weight = use_weights ? edge.weight : 1.0f0
        
        # Add the forward edge
        push!(origin_indices, edge.origin_index)
        push!(destination_indices, edge.destination_index)
        push!(weights, edge_weight)
        
        # Add reverse edge (all edges are bidirectional)
        push!(origin_indices, edge.destination_index)
        push!(destination_indices, edge.origin_index)
        push!(weights, edge_weight)
    end
    
    # Create sparse adjacency matrix
    A = sparse(origin_indices, destination_indices, weights, length(graph.nodes), length(graph.nodes))
  
    return A
end
