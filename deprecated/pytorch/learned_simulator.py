import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import Data

import graph_network
from connectivity_utils import compute_connectivity_for_batch

STD_EPSILON = 1e-8

def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

DEVICE = _get_device()

def time_diff(input_sequence: Tensor) -> Tensor:
    """Computes finite differences along the sequence dimension."""
    return input_sequence[:, 1:] - input_sequence[:, :-1]

class LearnedSimulator(nn.Module):
    """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

    def __init__(
        self,
        num_dimensions: int,
        connectivity_radius: float,
        graph_network_kwargs: dict,
        boundaries: list,
        normalization_stats: dict,
        num_particle_types: int,
        particle_type_embedding_size: int
    ):
        super().__init__()

        self._connectivity_radius = connectivity_radius
        self._num_particle_types = num_particle_types
        # Match TF: convert boundaries to tensor once at init
        self._boundaries = torch.tensor(boundaries, dtype=torch.float32, device=DEVICE)
        self._normalization_stats = normalization_stats

        self._graph_network = graph_network.EncodeProcessDecode(
            output_size=num_dimensions, **graph_network_kwargs)

        if self._num_particle_types > 1:
            # Match TF: initialize embedding as parameter instead of nn.Embedding
            self._particle_type_embedding = nn.Parameter(
                torch.empty(self._num_particle_types, particle_type_embedding_size, device=DEVICE))
            # Initialize using same approach as TF
            nn.init.uniform_(self._particle_type_embedding, -0.1, 0.1)
        else:
            self._particle_type_embedding = None

    def forward(
        self,
        position_sequence: Tensor,
        n_particles_per_example: Tensor,
        global_context: Tensor = None,
        particle_types: Tensor = None
    ) -> Tensor:
        input_graphs_tuple = self._encoder_preprocessor(
            position_sequence, n_particles_per_example, global_context, particle_types)

        normalized_acceleration = self._graph_network(input_graphs_tuple)
        next_position = self._decoder_postprocessor(normalized_acceleration, position_sequence)
        return next_position

    def _encoder_preprocessor(
        self,
        position_sequence: Tensor,
        n_node: Tensor,
        global_context: Tensor,
        particle_types: Tensor
    ) -> Data:
        # Match TF: Move tensors to same device at start
        device = self._boundaries.device
        position_sequence = position_sequence.to(device)
        n_node = n_node.to(device)
        if global_context is not None:
            global_context = global_context.to(device)
        if particle_types is not None:
            particle_types = particle_types.to(device)

        most_recent_position = position_sequence[:, -1]
        velocity_sequence = time_diff(position_sequence)

        # Get connectivity
        senders, receivers, n_edge = compute_connectivity_for_batch(
            positions=most_recent_position,
            n_node=n_node,
            radius=self._connectivity_radius,
            add_self_edges=True
        )

        # Node features
        node_features = []

        # Match TF normalization approach
        velocity_stats = self._normalization_stats["velocity"]
        normalized_velocity_sequence = (
            velocity_sequence - velocity_stats["mean"].to(device)
        ) / velocity_stats["std"].to(device)

        # Flatten velocity sequence matching TF's MergeDims behavior
        num_particles, seq_len_minus_one, dim = normalized_velocity_sequence.shape
        flat_velocity_sequence = normalized_velocity_sequence.reshape(num_particles, -1)
        node_features.append(flat_velocity_sequence)

        # Distance to boundaries, matching TF implementation
        distance_to_lower = most_recent_position - self._boundaries[:, 0]
        distance_to_upper = self._boundaries[:, 1] - most_recent_position
        distance_to_boundaries = torch.cat([distance_to_lower, distance_to_upper], dim=1)
        
        normalized_clipped_distance_to_boundaries = torch.clamp(
            distance_to_boundaries / self._connectivity_radius, min=-1., max=1.)
        node_features.append(normalized_clipped_distance_to_boundaries)

        # Particle type embedding, matching TF lookup behavior
        if self._num_particle_types > 1 and particle_types is not None:
            particle_type_embeddings = F.embedding(particle_types, self._particle_type_embedding)
            node_features.append(particle_type_embeddings)

        # Edge features
        edge_features = []

        # Relative displacements and distances
        sender_positions = most_recent_position[senders]
        receiver_positions = most_recent_position[receivers]
        normalized_relative_displacements = (
            sender_positions - receiver_positions) / self._connectivity_radius
        edge_features.append(normalized_relative_displacements)

        normalized_relative_distances = torch.norm(
            normalized_relative_displacements, dim=-1, keepdim=True)
        edge_features.append(normalized_relative_distances)

        # Global context normalization matching TF
        global_attr = None
        if global_context is not None:
            if "context" in self._normalization_stats:
                context_stats = self._normalization_stats["context"]
                std_clamped = torch.clamp_min(context_stats["std"].to(device), STD_EPSILON)
                global_attr = (
                    global_context - context_stats["mean"].to(device)) / std_clamped
            else:
                global_attr = global_context

        # Construct Data object
        x = torch.cat(node_features, dim=-1)
        edge_attr = torch.cat(edge_features, dim=-1)
        edge_index = torch.stack([senders, receivers], dim=0)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        if global_attr is not None:
            data.globals = global_attr

        data.n_node = n_node
        data.n_edge = n_edge

        return data

    def _decoder_postprocessor(
        self,
        normalized_acceleration: Tensor,
        position_sequence: Tensor
    ) -> Tensor:
        acceleration_stats = self._normalization_stats["acceleration"]
        # Match TF denormalization order
        acceleration = (
            normalized_acceleration * acceleration_stats["std"]
        ) + acceleration_stats["mean"]

        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = most_recent_position - position_sequence[:, -2]

        # Euler integration matching TF
        new_velocity = most_recent_velocity + acceleration  # dt=1
        new_position = most_recent_position + new_velocity  # dt=1
        return new_position

    def get_predicted_and_target_normalized_accelerations(
        self,
        next_position: Tensor,
        position_sequence_noise: Tensor,
        position_sequence: Tensor,
        n_particles_per_example: Tensor,
        global_context: Tensor = None,
        particle_types: Tensor = None
    ) -> (Tensor, Tensor):
        # Add noise to input sequence
        noisy_position_sequence = position_sequence + position_sequence_noise

        # Forward pass with noise
        input_graphs_tuple = self._encoder_preprocessor(
            noisy_position_sequence,
            n_particles_per_example,
            global_context,
            particle_types
        )
        predicted_normalized_acceleration = self._graph_network(input_graphs_tuple)

        # Adjust next_position as in TF implementation
        next_position_adjusted = next_position + position_sequence_noise[:, -1]
        target_normalized_acceleration = self._inverse_decoder_postprocessor(
            next_position_adjusted,
            noisy_position_sequence
        )

        return predicted_normalized_acceleration, target_normalized_acceleration

    def _inverse_decoder_postprocessor(
        self,
        next_position: Tensor,
        position_sequence: Tensor
    ) -> Tensor:
        # Match TF implementation exactly
        previous_position = position_sequence[:, -1]
        previous_velocity = previous_position - position_sequence[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity

        # Match TF normalization order
        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration = (
            acceleration - acceleration_stats["mean"]) / acceleration_stats["std"]
        return normalized_acceleration