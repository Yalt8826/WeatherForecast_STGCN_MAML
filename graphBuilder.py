"""
Graph construction utilities for spatial weather data.
"""
import torch
import numpy as np
from scipy.spatial import cKDTree


def build_spatial_graph(ds, k_neighbors=4):
    """
    Build k-NN spatial graph from lat/lon coordinates.
    
    Args:
        ds: xarray Dataset with 'latitude' and 'longitude' coordinates
        k_neighbors: Number of nearest neighbors to connect
    
    Returns:
        edge_index: Tensor of shape [2, num_edges] for PyTorch Geometric
        num_nodes: Total number of spatial nodes
        node_positions: Array of [lat, lon] positions
    """
    # Get lat/lon grids
    lats = ds.latitude.values
    lons = ds.longitude.values
    
    # Create meshgrid
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    # Flatten to get node positions
    node_positions = np.c_[lat_grid.ravel(), lon_grid.ravel()]
    num_nodes = len(node_positions)
    
    # Build k-NN graph using KDTree
    tree = cKDTree(node_positions)
    _, neighbors = tree.query(node_positions, k=k_neighbors+1)  # +1 includes self
    
    # Create edge list
    edges = []
    for node_idx, neighbor_indices in enumerate(neighbors):
        for neighbor_idx in neighbor_indices[1:]:  # Skip self-connection
            edges.append([node_idx, neighbor_idx])
    
    # Convert to tensor format [2, num_edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    print(f"Graph created: {num_nodes} nodes, {edge_index.shape[1]} edges")
    return edge_index, num_nodes, node_positions


def build_distance_weighted_graph(ds, distance_threshold=5.0):
    """
    Build graph with edges weighted by inverse distance.
    
    Args:
        ds: xarray Dataset
        distance_threshold: Maximum distance (degrees) for connections
    
    Returns:
        edge_index, edge_weights, num_nodes
    """
    lats = ds.latitude.values
    lons = ds.longitude.values
    
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    node_positions = np.c_[lat_grid.ravel(), lon_grid.ravel()]
    num_nodes = len(node_positions)
    
    edges = []
    weights = []
    
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            dist = np.linalg.norm(node_positions[i] - node_positions[j])
            if dist < distance_threshold and dist > 0:
                edges.append([i, j])
                edges.append([j, i])  # Undirected
                weight = 1.0 / dist
                weights.append(weight)
                weights.append(weight)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_weights = torch.tensor(weights, dtype=torch.float32)
    
    return edge_index, edge_weights, num_nodes
