import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_add
from typing import Callable, Optional


class MLP(nn.Module):
    """
    A simple MLP that mimics the behavior of Sonnet's snt.nets.MLP.
    It supports multiple hidden layers with ReLU activations and a final linear layer.
    """
    def __init__(self, 
                 input_size: int,
                 hidden_size: int, 
                 num_hidden_layers: int, 
                 output_size: int):
        super().__init__()
        layers = []
        current_size = input_size
        
        # Initialize hidden layers with ReLU activation
        for _ in range(num_hidden_layers):
            linear = nn.Linear(current_size, hidden_size)
            # TF-like initialization
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            current_size = hidden_size
            
        # Output layer (no activation)
        output_layer = nn.Linear(current_size, output_size)
        nn.init.xavier_uniform_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def build_mlp_with_layer_norm(input_size: int, hidden_size: int, num_hidden_layers: int, output_size: int) -> nn.Module:
    """
    Builds an MLP followed by a LayerNorm, mirroring the TF code
    """
    mlp = MLP(input_size, hidden_size, num_hidden_layers, output_size)
    layer_norm = nn.LayerNorm(output_size)
    # Initialize LayerNorm parameters
    nn.init.ones_(layer_norm.weight)
    nn.init.zeros_(layer_norm.bias)
    return nn.Sequential(mlp, layer_norm)


class GraphIndependent(nn.Module):
    """
    Applies independent transforms to edges, nodes, and globals.
    If certain attributes (edge_attr, globals) are None, it skips those transforms.
    """
    def __init__(self,
                 edge_model_fn: Optional[Callable[[], nn.Module]] = None,
                 node_model_fn: Optional[Callable[[], nn.Module]] = None,
                 global_model_fn: Optional[Callable[[], nn.Module]] = None):
        super().__init__()
        self.edge_model = edge_model_fn() if edge_model_fn is not None else None
        self.node_model = node_model_fn() if node_model_fn is not None else None
        self.global_model = global_model_fn() if global_model_fn is not None else None

    def forward(self, data):
        new_data = data.clone()

        # If we have an edge model and edge attributes, apply it
        if self.edge_model is not None and new_data.edge_attr is not None:
            new_data.edge_attr = self.edge_model(new_data.edge_attr)

        # If we have a node model, apply it
        if self.node_model is not None:
            new_data.x = self.node_model(new_data.x)

        # If we have a global model and globals, apply it
        if self.global_model is not None and getattr(new_data, 'globals', None) is not None:
            new_data.globals = self.global_model(new_data.globals)

        return new_data


class InteractionNetwork(nn.Module):
    """
    Single step of message passing:
    - Updates edges using an edge model (if present).
    - Aggregates updated edges at receiving nodes.
    - Updates nodes using a node model.
    """
    def __init__(self,
                 edge_model_fn: Optional[Callable[[], nn.Module]],
                 node_model_fn: Callable[[], nn.Module],
                 reducer: Callable[[Tensor, Tensor, int], Tensor] = scatter_add):
        super().__init__()
        self.edge_model = edge_model_fn() if edge_model_fn is not None else None
        self.node_model = node_model_fn()
        self.reducer = reducer

    def forward(self, data):
        new_data = data.clone()
        senders = data.edge_index[0]
        receivers = data.edge_index[1]

        sender_features = data.x[senders]
        receiver_features = data.x[receivers]

        # If we have edge features and an edge model:
        if data.edge_attr is not None and self.edge_model is not None:
            edge_inputs = torch.cat([sender_features, receiver_features, data.edge_attr], dim=-1)
            updated_edge_attr = self.edge_model(edge_inputs)
        else:
            # If no edge features or no edge model:
            # JUST use sender and receiver features
            edge_inputs = torch.cat([sender_features, receiver_features], dim=-1)
            if self.edge_model is not None:
                updated_edge_attr = self.edge_model(edge_inputs)
            else:
                # No edge model at all, messages are just node pair features
                updated_edge_attr = edge_inputs

        num_nodes = data.x.size(0)
        aggregated_messages = self.reducer(
            updated_edge_attr,
            receivers,
            dim=0,
            dim_size=num_nodes
        )

        # Node model input: old node features + aggregated messages
        node_inputs = torch.cat([data.x, aggregated_messages], dim=-1)
        updated_nodes = self.node_model(node_inputs)

        new_data.x = updated_nodes
        new_data.edge_attr = updated_edge_attr
        return new_data


class EncodeProcessDecode(nn.Module):
    """
    PyTorch version of Encode-Process-Decode, closely following TF code:
    - Uses an encoder (GraphIndependent) to transform raw inputs into latent embeddings.
    - Uses multiple steps of InteractionNetwork-based message passing.
    - Decodes the latent node representations into outputs.
    """
    def __init__(self,
                 latent_size: int,
                 mlp_hidden_size: int,
                 mlp_num_hidden_layers: int,
                 num_message_passing_steps: int,
                 output_size: int,
                 reducer: Callable[[Tensor, Tensor, int], Tensor] = scatter_add):
        super().__init__()
        self._latent_size = latent_size
        self._mlp_hidden_size = mlp_hidden_size
        self._mlp_num_hidden_layers = mlp_num_hidden_layers
        self._num_message_passing_steps = num_message_passing_steps
        self._output_size = output_size
        self._reducer = reducer

        self.node_in_dim = None
        self.edge_in_dim = None
        self.global_in_dim = 0

        self._encoder_network = None
        self._processor_networks = nn.ModuleList()
        self._decoder_network = None

    def _broadcast_globals_to_nodes(self, data):
        """Broadcasts global features to all nodes."""
        if hasattr(data, 'globals') and data.globals is not None:
            num_nodes = data.x.size(0)
            return data.globals.expand(num_nodes, -1)
        return None

    def initialize_encoder(self, node_in_dim: int, edge_in_dim: int, global_in_dim: int = 0):
        """
        Initializes the encoder and processor networks once the input dimensions are known.
        This MUST be called before the first forward pass.
        """
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.global_in_dim = global_in_dim

        # If no edge features, we skip edge_model in the encoder
        if self.edge_in_dim > 0:
            edge_model_fn = lambda: build_mlp_with_layer_norm(
                self.edge_in_dim, self._mlp_hidden_size, self._mlp_num_hidden_layers, self._latent_size
            )
        else:
            edge_model_fn = None

        # Node encoder always needed
        node_model_fn = lambda: build_mlp_with_layer_norm(
            self.node_in_dim + self.global_in_dim, self._mlp_hidden_size, self._mlp_num_hidden_layers, self._latent_size
        )

        self._encoder_network = GraphIndependent(
            edge_model_fn=edge_model_fn,
            node_model_fn=node_model_fn,
            global_model_fn=None
        )

        # Processor networks
        # After encoding, edges and nodes have dimension latent_size
        for _ in range(self._num_message_passing_steps):
            if self._encoder_network.edge_model is not None:
                # Edges have latent_size: so input is sender+receiver+edge = 3*latent_size
                edge_input_dim = 3 * self._latent_size
                processor_edge_model_fn = lambda: build_mlp_with_layer_norm(
                    edge_input_dim, self._mlp_hidden_size, self._mlp_num_hidden_layers, self._latent_size
                )
            else:
                # No edge model in the encoder means no edge latent from encoding
                # Edge input dim = sender+receiver only = 2*latent_size
                processor_edge_model_fn = lambda: build_mlp_with_layer_norm(
                    2 * self._latent_size, self._mlp_hidden_size, self._mlp_num_hidden_layers, self._latent_size
                )

            node_input_dim = 2 * self._latent_size  # Current + aggregated
            processor_node_model_fn = lambda: build_mlp_with_layer_norm(
                node_input_dim, self._mlp_hidden_size, self._mlp_num_hidden_layers, self._latent_size
            )

            self._processor_networks.append(
                InteractionNetwork(
                    edge_model_fn=processor_edge_model_fn if (edge_model_fn is not None) else None,
                    node_model_fn=processor_node_model_fn,
                    reducer=self._reducer
                )
            )

        # Decoder: from latent_size to output_size
        self._decoder_network = MLP(
            input_size=self._latent_size,
            hidden_size=self._mlp_hidden_size,
            num_hidden_layers=self._mlp_num_hidden_layers,
            output_size=self._output_size
        )

    def forward(self, data):
        if self._encoder_network is None:
            raise RuntimeError("Encoder not initialized. Call initialize_encoder(...) before forward.")

        # Encode
        latent_graph_0 = self._encode(data)

        # Process
        latent_graph_m = self._process(latent_graph_0)

        # Decode
        return self._decode(latent_graph_m)

    def _encode(self, data):
        # Broadcast globals to nodes if present
        broadcasted_globals = self._broadcast_globals_to_nodes(data)
        if broadcasted_globals is not None:
            data.x = torch.cat([data.x, broadcasted_globals], dim=-1)
            data.globals = None

        latent_graph_0 = self._encoder_network(data)
        return latent_graph_0

    def _process(self, latent_graph_0):
        latent_graph_prev_k = latent_graph_0
        latent_graph_k = latent_graph_0
        for processor_network_k in self._processor_networks:
            latent_graph_k = self._process_step(processor_network_k, latent_graph_prev_k)
            latent_graph_prev_k = latent_graph_k
        return latent_graph_k

    def _process_step(self, processor_network_k, latent_graph_prev_k):
        latent_graph_k = processor_network_k(latent_graph_prev_k)
        # Residual connections
        latent_graph_k.x = latent_graph_k.x + latent_graph_prev_k.x
        if latent_graph_k.edge_attr is not None and latent_graph_prev_k.edge_attr is not None:
            latent_graph_k.edge_attr = latent_graph_k.edge_attr + latent_graph_prev_k.edge_attr
        return latent_graph_k

    def _decode(self, latent_graph):
        return self._decoder_network(latent_graph.x)