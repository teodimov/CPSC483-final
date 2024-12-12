import torch
import torch.nn as nn
from torch_scatter import scatter
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class GraphData:
    """Container for graph data."""
    node_feat: torch.Tensor  # [N, F_x] Node features
    edge_index: torch.Tensor  # [2, E] Edge indices
    edge_feat: torch.Tensor  # [E, F_e] Edge features

class EdgeModel(nn.Module):
    """Updates edge features based on source and target node features."""
    def __init__(self, edge_mlp: nn.Module):
        """
        Args:
            edge_mlp: MLP that processes concatenated source node, target node, and edge features
        """
        super().__init__()
        self.edge_mlp = edge_mlp

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, F_x], where N is the number of nodes
            edge_index: Edge connectivity [2, E] with max entry N - 1
            edge_attr: Edge features [E, F_e]
        
        Returns:
            Updated edge features [E, F_e_out]
        """
        edge_index = edge_index.permute(1, 0)  # [E, 2]
        sender_features = x[edge_index[0]]     # Source node features
        receiver_features = x[edge_index[1]]   # Target node features
        
        # Concatenate all features for edge update
        edge_inputs = torch.cat([
            sender_features, 
            receiver_features, 
            edge_attr
        ], dim=-1)
        
        return self.edge_mlp(edge_inputs)

class NodeModel(nn.Module):
    """Updates node features based on aggregated edge features."""
    def __init__(self, node_mlp: nn.Module):
        """
        Args:
            node_mlp: MLP that processes concatenated node and aggregated edge features
        """
        super().__init__()
        self.node_mlp = node_mlp

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, F_x], where N is the number of nodes
            edge_index: Edge connectivity [2, E] with max entry N - 1
            edge_attr: Edge features [E, F_e]
        
        Returns:
            Updated node features [N, F_x_out]
        """
        num_nodes = x.size(0)
        
        # Aggregate incoming edge features for each node
        aggregated_edges = scatter(
            edge_attr,              # Edge features to aggregate
            edge_index[1],          # Target node indices
            dim=0,                  # Dimension to aggregate along
            dim_size=num_nodes,     # Number of nodes
            reduce='mean'           # Aggregation function
        )
        
        # Concatenate node features with aggregated edge features
        node_inputs = torch.cat([x, aggregated_edges], dim=-1)
        
        return self.node_mlp(node_inputs)

class GraphNetwork(nn.Module):
    """Complete graph neural network with edge and node updates."""
    def __init__(self, edge_model: Optional[EdgeModel] = None, 
                 node_model: Optional[NodeModel] = None):
        """
        Args:
            edge_model: Optional model for updating edge features
            node_model: Optional model for updating node features
        """
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.reset_parameters()

    def reset_parameters(self):
        """Resets parameters of all sub-models if they implement reset_parameters."""
        for model in [self.node_model, self.edge_model]:
            if hasattr(model, 'reset_parameters'):
                model.reset_parameters()

    def forward(self, graph: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            graph: Dictionary containing:
                - node_feat: Node features [N, F_x]
                - edge_index: Edge connectivity [2, E]
                - edge_feat: Edge features [E, F_e]
        
        Returns:
            Tuple of:
                - Updated node features [N, F_x_out]
                - Updated edge features [E, F_e_out]
        """
        x = graph['node_feat']
        edge_index = graph['edge_index']
        edge_attr = graph['edge_feat']
        
        # Update edge features if edge model exists
        if self.edge_model is not None:
            edge_attr = self.edge_model(x, edge_index, edge_attr)

        # Update node features if node model exists
        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr)

        return x, edge_attr
