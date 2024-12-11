import numpy as np
from sklearn import neighbors
import torch
from functools import lru_cache
from typing import Tuple, Union, Optional

@lru_cache(maxsize=128)
def _compute_connectivity_cached(
    positions_tuple: Tuple[Tuple[float, ...], ...],
    radius: float,
    add_self_edges: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cached version of connectivity computation. Takes tuple of tuples for hashability.
    """
    positions = np.array(positions_tuple)
    tree = neighbors.KDTree(positions)
    receivers_list = tree.query_radius(positions, r=radius)
    num_nodes = len(positions)
    
    senders = np.repeat(np.arange(num_nodes), [len(a) for a in receivers_list])
    receivers = np.concatenate(receivers_list, axis=0)
    
    if not add_self_edges:
        mask = (senders != receivers)
        senders = senders[mask]
        receivers = receivers[mask]
        
    return senders, receivers

def _compute_connectivity(
    positions: np.ndarray, 
    radius: float, 
    add_self_edges: bool,
    use_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the indices of connected edges with radius connectivity.

    Args:
        positions: Positions of nodes in the graph. Shape: [num_nodes_in_graph, num_dims].
        radius: Radius of connectivity.
        add_self_edges: Whether to include self edges or not.
        use_cache: Whether to use caching for repeated position configurations.

    Returns:
        Tuple of:
            senders indices [num_edges_in_graph]
            receivers indices [num_edges_in_graph]
    """
    if use_cache:
        # Convert positions to tuple of tuples for hashability
        positions_tuple = tuple(tuple(row) for row in positions)
        return _compute_connectivity_cached(positions_tuple, radius, add_self_edges)
    
    tree = neighbors.KDTree(positions)
    receivers_list = tree.query_radius(positions, r=radius)
    num_nodes = len(positions)
    
    senders = np.repeat(np.arange(num_nodes), [len(a) for a in receivers_list])
    receivers = np.concatenate(receivers_list, axis=0)
    
    if not add_self_edges:
        mask = (senders != receivers)
        senders = senders[mask]
        receivers = receivers[mask]
        
    return senders, receivers

def _compute_connectivity_for_batch(
    positions: np.ndarray,
    n_node: np.ndarray,
    radius: float,
    add_self_edges: bool,
    use_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute connectivity for a batch of graphs.

    Args:
        positions: Positions of nodes in the batch of graphs. Shape:
            [num_nodes_in_batch, num_dims].
        n_node: Number of nodes for each graph in the batch. Shape: [num_graphs_in_batch].
        radius: Radius of connectivity.
        add_self_edges: Whether to include self edges or not.
        use_cache: Whether to use caching for repeated position configurations.

    Returns:
        Tuple of:
            senders: [num_edges_in_batch]
            receivers: [num_edges_in_batch]
            n_edge: number of edges per graph [num_graphs_in_batch]
    """
    # Split positions into list for each graph, following TF implementation
    positions_per_graph_list = np.split(positions, np.cumsum(n_node[:-1]), axis=0)
    
    receivers_list = []
    senders_list = []
    n_edge_list = []
    num_nodes_in_previous_graphs = 0
    
    # Process each graph in the batch
    for positions_graph_i in positions_per_graph_list:
        senders_graph_i, receivers_graph_i = _compute_connectivity(
            positions_graph_i, radius, add_self_edges, use_cache
        )
        
        num_edges_graph_i = len(senders_graph_i)
        n_edge_list.append(num_edges_graph_i)
        
        # Adjust indices by offset from previous graphs
        receivers_list.append(receivers_graph_i + num_nodes_in_previous_graphs)
        senders_list.append(senders_graph_i + num_nodes_in_previous_graphs)
        
        num_nodes_in_previous_graphs += len(positions_graph_i)
    
    # Concatenate results, ensuring int32 dtype as in TF version
    senders = np.concatenate(senders_list, axis=0).astype(np.int32)
    receivers = np.concatenate(receivers_list, axis=0).astype(np.int32)
    n_edge = np.stack(n_edge_list).astype(np.int32)
    
    return senders, receivers, n_edge

def compute_connectivity_for_batch(
    positions: Union[torch.Tensor, np.ndarray],
    n_node: Union[torch.Tensor, np.ndarray],
    radius: float,
    add_self_edges: bool = True,
    use_cache: bool = True,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch-compatible version of connectivity computation.
    
    Args:
        positions: Node positions. Shape: [num_nodes_in_batch, num_dims].
        n_node: Number of nodes per graph. Shape: [num_graphs_in_batch].
        radius: Radius of connectivity.
        add_self_edges: Whether to include self edges.
        use_cache: Whether to use caching for repeated configurations.
        device: Target device for output tensors. If None, uses positions' device.

    Returns:
        Tuple of:
            senders: Shape [num_edges_in_batch]
            receivers: Shape [num_edges_in_batch]
            n_edge: Shape [num_graphs_in_batch]
    """
    # Convert inputs to numpy if needed
    positions_np = positions.detach().cpu().numpy() if torch.is_tensor(positions) else positions
    n_node_np = n_node.detach().cpu().numpy() if torch.is_tensor(n_node) else n_node
    
    # Compute connectivity
    senders_np, receivers_np, n_edge_np = _compute_connectivity_for_batch(
        positions_np, n_node_np, radius, add_self_edges, use_cache
    )
    
    # Determine device
    if device is None and torch.is_tensor(positions):
        device = positions.device
    elif device is None:
        device = torch.device('cpu')
    
    # Convert to torch tensors and move to correct device
    senders = torch.from_numpy(senders_np).long().to(device)
    receivers = torch.from_numpy(receivers_np).long().to(device)
    n_edge = torch.from_numpy(n_edge_np).long().to(device)
    
    return senders, receivers, n_edge