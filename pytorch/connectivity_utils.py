# connectivity_utils.py contains functions to compute connectivity between nodes in a graph.

import numpy as np
from sklearn import neighbors
import torch

def _compute_connectivity(
    positions: np.ndarray, 
    radius: float, 
    add_self_edges: bool
) -> (np.ndarray, np.ndarray):
    """
    Get the indices of connected edges with radius connectivity.

    Args:
        positions: Positions of nodes in the graph. Shape: [num_nodes_in_graph, num_dims].
        radius: Radius of connectivity.
        add_self_edges: Whether to include self edges or not.

    Returns:
        senders indices [num_edges_in_graph]
        receivers indices [num_edges_in_graph]
    """
    # KDTree for querying neighbors
    tree = neighbors.KDTree(positions)
    # Query neighbors within radius
    receivers_list = tree.query_radius(positions, r=radius)
    num_nodes = len(positions)

    # senders and receivers
    senders = np.repeat(np.arange(num_nodes), [len(a) for a in receivers_list])
    receivers = np.concatenate(receivers_list, axis=0)

    if not add_self_edges:
        # remove self edges
        mask = (senders != receivers)
        senders = senders[mask]
        receivers = receivers[mask]

    return senders, receivers


def _compute_connectivity_for_batch(
    positions: np.ndarray,
    n_node: np.ndarray,
    radius: float,
    add_self_edges: bool
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    `_compute_connectivity` for a batch of graphs.

    Args:
        positions: Positions of nodes in the batch of graphs. Shape:
            [num_nodes_in_batch, num_dims].
        n_node: Number of nodes for each graph in the batch. Shape: [num_graphs_in_batch].
        radius: Radius of connectivity.
        add_self_edges: Whether to include self edges or not.

    Returns:
        senders: [num_edges_in_batch]
        receivers: [num_edges_in_batch]
        n_edge: number of edges per graph [num_graphs_in_batch]
    """

    # Split positions into a list, ONE for each graph
    positions_per_graph_list = np.split(positions, np.cumsum(n_node[:-1]), axis=0)

    receivers_list = []
    senders_list = []
    n_edge_list = []
    num_nodes_in_previous_graphs = 0

    # connectivity for each graph
    for positions_graph_i in positions_per_graph_list:
        senders_graph_i, receivers_graph_i = _compute_connectivity(
            positions_graph_i, radius, add_self_edges
        )

        num_edges_graph_i = len(senders_graph_i)
        n_edge_list.append(num_edges_graph_i)

        # adjust indices by offset of prev graphs
        receivers_list.append(receivers_graph_i + num_nodes_in_previous_graphs)
        senders_list.append(senders_graph_i + num_nodes_in_previous_graphs)

        num_nodes_graph_i = len(positions_graph_i)
        num_nodes_in_previous_graphs += num_nodes_graph_i

    # concat results
    senders = np.concatenate(senders_list, axis=0).astype(np.int32)
    receivers = np.concatenate(receivers_list, axis=0).astype(np.int32)
    n_edge = np.stack(n_edge_list).astype(np.int32)

    return senders, receivers, n_edge


def compute_connectivity_for_batch(
    positions: torch.Tensor,
    n_node: torch.Tensor,
    radius: float,
    add_self_edges: bool = True
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    PyTorch-compatible version of `compute_connectivity_for_batch_pyfunc`.

    Args:
        positions: (torch.Tensor) Positions of nodes. Shape: [num_nodes_in_batch, num_dims].
        n_node: (torch.Tensor) Number of nodes in each graph. Shape: [num_graphs_in_batch].
        radius: (float) Radius of connectivity.
        add_self_edges: (bool) Whether to include self edges or not.

    Returns:
        senders (torch.Tensor): Shape [num_edges_in_batch]
        receivers (torch.Tensor): Shape [num_edges_in_batch]
        n_edge (torch.Tensor): Shape [num_graphs_in_batch]
    """

    # numpy needed for KDTree computations
    positions_np = positions.detach().cpu().numpy() if isinstance(positions, torch.Tensor) else positions
    n_node_np = n_node.detach().cpu().numpy() if isinstance(n_node, torch.Tensor) else n_node

    senders_np, receivers_np, n_edge_np = _compute_connectivity_for_batch(
        positions_np, n_node_np, radius, add_self_edges
    )

    # revert to torch long tensors
    senders = torch.from_numpy(senders_np).long()
    receivers = torch.from_numpy(receivers_np).long()
    n_edge = torch.from_numpy(n_edge_np).long()

    return senders, receivers, n_edge
