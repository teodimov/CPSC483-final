from typing import Tuple
import numpy as np
import torch
from sklearn.neighbors import KDTree
from typing import Union

def _compute_connectivity(
    positions: np.ndarray,
    radius: Union[float, np.ndarray],
    add_self_edges: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the indices of connected edges with radius connectivity.

    Args:
        positions: Node positions in the graph. [num_nodes, num_dims]
        radius: Connectivity radius (scalar or array of shape [num_nodes]).
        add_self_edges: Whether to include self edges or not.

    Returns:
        Tuple[np.ndarray, np.ndarray]: senders and receivers indices of edges.
    """
    print(f"Positions shape: {positions.shape}, dtype: {positions.dtype}")
    print(f"Radius type: {type(radius)}")
    if isinstance(radius, np.ndarray):
        print(f"Radius shape: {radius.shape}, dtype: {radius.dtype}")
        print(f"Radius values: min={radius.min()}, max={radius.max()}")
    elif isinstance(radius, (int, float)):
        print(f"Radius scalar value: {radius}")
        assert radius > 0, "Radius must be a positive scalar."

    # If radius is a PyTorch tensor, convert it to NumPy array
    if isinstance(radius, torch.Tensor):
        print("Converting radius from PyTorch tensor to NumPy array.")
        radius = radius.cpu().numpy()

    # Validate radius shape

    if isinstance(radius, np.ndarray):
        if radius.ndim == 1 and radius.shape[0] == positions.shape[0]:
            print("Radius validated successfully.")
        else:
            raise ValueError(
                f"Radius must be scalar or 1D array of shape ({positions.shape[0],}), "
                f"but got shape {radius.shape}."
            )

    # Initialize KDTree
    tree = KDTree(positions)

    # Query radius
    try:
        receivers_list = tree.query_radius(positions, r=radius)
    except Exception as e:
        print(f"Error in query_radius with positions shape {positions.shape} and radius {radius}")
        raise e

    # Generate sender and receiver indices
    num_nodes = positions.shape[0]
    senders = np.repeat(np.arange(num_nodes), [len(a) for a in receivers_list])
    receivers = np.concatenate(receivers_list, axis=0)

    # Remove self-edges if specified
    if not add_self_edges:
        mask = senders != receivers
        senders = senders[mask]
        receivers = receivers[mask]

    return senders, receivers



def compute_connectivity_for_batch(
    positions: np.ndarray, 
    n_node: np.ndarray,
    radius: float,
    add_self_edges: bool = True,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`compute_connectivity` for a batch of graphs.
    Args:
        positions: [num_nodes_in_batch, num_dims]
        n_node: number of nodes for each graph in the batch. [num_graphs_in_batch]
        radius: Radius of connectivity.
        add_self_edges: Whether to include self edges or not.
    Returns:
        sender indices [num_edges_in_batch]
        receiver indices [num_edges_in_batch]
        number of edges per graph [num_graphs_in_batch]
    """
    n_graphs = len(n_node)
    receivers_list = []
    senders_list = []
    n_edge_list = np.empty(n_graphs, dtype=np.int32)
    
    positions_per_graph_list = np.split(positions, np.cumsum(n_node[:-1]), axis=0)
    num_nodes_in_previous_graphs = 0

    # Compute connectivity for each graph in the batch.
    for i, positions_graph_i in enumerate(positions_per_graph_list):
        senders_graph_i, receivers_graph_i = _compute_connectivity(
            positions_graph_i, radius, add_self_edges)
        
        n_edge_list[i] = len(senders_graph_i)

        receivers_list.append(receivers_graph_i + num_nodes_in_previous_graphs)
        senders_list.append(senders_graph_i + num_nodes_in_previous_graphs)

        num_nodes_in_previous_graphs += positions_graph_i.shape[0]

    # Concatenate all of the results.
    senders = torch.tensor(np.concatenate(senders_list, axis=0),
                          dtype=torch.int64, device=device)
    receivers = torch.tensor(np.concatenate(receivers_list, axis=0),
                           dtype=torch.int64, device=device)
    n_edge = torch.tensor(n_edge_list, dtype=torch.int64, device=device)

    return senders, receivers, n_edge
