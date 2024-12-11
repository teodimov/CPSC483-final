# learned_simulator.py: PyTorch version of the LearnedSimulator class.

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
    """
    Computes finite differences along the sequence dimension.
    input_sequence: [num_particles, sequence_length, num_dimensions]
    returns: [num_particles, sequence_length-1, num_dimensions]
    """
    return input_sequence[:, 1:] - input_sequence[:, :-1]

class LearnedSimulator(nn.Module):
    """
    PyTorch version of the LearnedSimulator class.
    This model predicts the next position of particles given a sequence of their previous positions.
    """

    def __init__(
        self,
        num_dimensions: int,
        connectivity_radius: float,
        graph_network_kwargs,
        boundaries,
        normalization_stats: dict,
        num_particle_types: int,
        particle_type_embedding_size: int
    ):
        """
        Args:
          num_dimensions: Dimensionality of the problem.
          connectivity_radius: Radius of connectivity for building edges.
          graph_network: The graph network model (EncodeProcessDecode) instance.
          boundaries: List of tuples (lower_bound, upper_bound) for each dimension.
          normalization_stats: Dict with "acceleration", "velocity", and optionally "context" stats (mean, std).
          num_particle_types: Number of different particle types.
          particle_type_embedding_size: Embedding size for the particle type.
        """
        super().__init__()

        self._connectivity_radius = connectivity_radius
        self._num_particle_types = num_particle_types
        self._boundaries = torch.tensor(boundaries, dtype=torch.float32, device=DEVICE)  # Shape: [num_dimensions, 2]
        self._normalization_stats = normalization_stats

        self._graph_network = graph_network.EncodeProcessDecode(
          output_size=num_dimensions, **graph_network_kwargs)

        if self._num_particle_types > 1:
            self._particle_type_embedding = nn.Embedding(self._num_particle_types, particle_type_embedding_size)
        else:
            self._particle_type_embedding = None

        self.num_dimensions = num_dimensions

    def forward(
        self,
        position_sequence: Tensor,
        n_particles_per_example: Tensor,
        global_context: Tensor = None,
        particle_types: Tensor = None
    ) -> Tensor:
        """
        Produce the next position for each particle.

        Args:
          position_sequence: [num_particles_in_batch, sequence_length, num_dimensions]
          n_particles_per_example: [batch_size] with number of particles per graph.
          global_context: [batch_size, context_size] or None
          particle_types: [num_particles_in_batch] int tensor with particle types.

        Returns:
          next_position: [num_particles_in_batch, num_dimensions]
        """
        input_graphs_tuple = self._encoder_preprocessor(
            position_sequence, n_particles_per_example, global_context, particle_types
        )

        # Pass the graph through the model
        normalized_acceleration = self._graph_network(input_graphs_tuple)

        # Decode back to positions
        next_position = self._decoder_postprocessor(normalized_acceleration, position_sequence)
        return next_position

    def _encoder_preprocessor(
        self,
        position_sequence: Tensor,
        n_node: Tensor,
        global_context: Tensor,
        particle_types: Tensor
    ) -> Data:
        """
        Prepares input graphs for the graph network.

        Args:
          position_sequence: [num_particles, sequence_length, num_dimensions]
          n_node: [batch_size]
          global_context: [batch_size, context_size] or None
          particle_types: [num_particles] or None

        Returns:
          data (PyG Data object) containing:
            - x: node features
            - edge_attr: edge features
            - edge_index: [2, num_edges]
            - global_attr: global features (if any)
            - n_node, n_edge stored as attributes
        """
        # put tensors on SAME device
        position_sequence = position_sequence.to(self._boundaries.device)
        n_node = n_node.to(self._boundaries.device)
        if global_context is not None:
            global_context = global_context.to(self._boundaries.device)
        if particle_types is not None:
            particle_types = particle_types.to(self._boundaries.device)
            
        # Most recent positions and velocities
        most_recent_position = position_sequence[:, -1]  # [num_particles, num_dimensions]
        velocity_sequence = time_diff(position_sequence) # [num_particles, seq_length-1, num_dimensions]

        # Compute connectivity
        senders, receivers, n_edge = compute_connectivity_for_batch(
            positions=most_recent_position,
            n_node=n_node,
            radius=self._connectivity_radius,
            add_self_edges=True
        )

        # Node features
        node_features = []

        # Normalize velocity sequence
        # velocity_stats: mean and std
        velocity_stats = self._normalization_stats["velocity"]
        # shape: velocity_sequence [num_particles, seq_length-1, num_dimensions]
        normalized_velocity_sequence = (velocity_sequence - velocity_stats["mean"].to(self._boundaries.device)) / velocity_stats["std"].to(self._boundaries.device)

        # Flatten velocity sequence over time
        # Originally used snt.MergeDims; we just reshape:
        # shape: [num_particles, (seq_length-1)*num_dimensions]
        num_particles, seq_len_minus_one, dim = normalized_velocity_sequence.shape
        flat_velocity_sequence = normalized_velocity_sequence.reshape(num_particles, -1)
        node_features.append(flat_velocity_sequence)

        # Distance to boundaries
        # boundaries: [num_dimensions, 2]
        # distance_to_lower: [num_particles, num_dimensions]
        distance_to_lower_boundary = most_recent_position - self._boundaries[:, 0].to(self._boundaries.device)
        # distance_to_upper: [num_particles, num_dimensions]
        distance_to_upper_boundary = self._boundaries[:, 1].to(self._boundaries.device) - most_recent_position

        # Concatenate distances: shape [num_particles, 2*num_dimensions]
        distance_to_boundaries = torch.cat([distance_to_lower_boundary, distance_to_upper_boundary], dim=1)

        # Normalize distances by radius and clip
        normalized_clipped_distance_to_boundaries = torch.clamp(
            distance_to_boundaries / self._connectivity_radius, min=-1., max=1.
        )
        node_features.append(normalized_clipped_distance_to_boundaries)

        # Particle type embedding
        if self._num_particle_types > 1 and particle_types is not None:
            particle_type_embeddings = self._particle_type_embedding(particle_types)  # [num_particles, embedding_size]
            node_features.append(particle_type_embeddings)

        # Edge features
        edge_features = []

        # Relative displacements normalized by radius
        # senders, receivers: [num_edges]
        # gather node positions for senders and receivers
        sender_positions = most_recent_position[senders]
        receiver_positions = most_recent_position[receivers]
        normalized_relative_displacements = (sender_positions - receiver_positions) / self._connectivity_radius
        edge_features.append(normalized_relative_displacements)

        # Norm of relative displacements
        normalized_relative_distances = torch.norm(normalized_relative_displacements, dim=-1, keepdim=True)
        edge_features.append(normalized_relative_distances)

        # Normalize global context if exists
        global_attr = None
        if global_context is not None:
            # context_stats may not always be available
            if "context" in self._normalization_stats:
                context_stats = self._normalization_stats["context"]
                # Avoid division by zero
                std_clamped = torch.clamp_min(context_stats["std"].to(self._boundaries.device), STD_EPSILON)
                global_attr = (global_context - context_stats["mean"].to(self._boundaries.device)) / std_clamped
            else:
                global_attr = global_context

        # Construct the Data object
        x = torch.cat(node_features, dim=-1)  # [num_particles, x_dim]
        edge_attr = torch.cat(edge_features, dim=-1)  # [num_edges, edge_dim]

        edge_index = torch.stack([senders, receivers], dim=0)  # [2, num_edges]

        data = Data(
            x=x.to(DEVICE), 
            edge_index=edge_index.to(DEVICE), 
            edge_attr=edge_attr.to(DEVICE)
        )

        if global_attr is not None:
            data.globals = global_attr.to(DEVICE)

        # Store n_node, n_edge as attributes if needed
        data.n_node = n_node
        data.n_edge = n_edge

        return data

    def _decoder_postprocessor(
        self,
        normalized_acceleration: Tensor,
        position_sequence: Tensor
    ) -> Tensor:
        """
        Convert normalized acceleration back to positions using Euler integration.

        Args:
          normalized_acceleration: [num_particles, num_dimensions]
          position_sequence: [num_particles, sequence_length, num_dimensions]

        Returns:
          next_position: [num_particles, num_dimensions]
        """
        acceleration_stats = self._normalization_stats["acceleration"]
        # acceleration: denormalize
        acceleration = normalized_acceleration * acceleration_stats["std"] + acceleration_stats["mean"]

        # Euler step
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = most_recent_position - position_sequence[:, -2]

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
        """
        Compute predicted and target normalized accelerations for training.

        Args:
          next_position: [num_particles, num_dimensions] ground-truth next position
          position_sequence_noise: same shape as position_sequence, additive noise.
          position_sequence: [num_particles, seq_length, num_dimensions]
          n_particles_per_example: [batch_size]
          global_context: [batch_size, context_size] or None
          particle_types: [num_particles] or None

        Returns:
          predicted_normalized_acceleration: [num_particles, num_dimensions]
          target_normalized_acceleration: [num_particles, num_dimensions]
        """

        # Add noise to input position sequence
        noisy_position_sequence = position_sequence + position_sequence_noise

        # Forward pass with noisy inputs
        input_graphs_tuple = self._encoder_preprocessor(
            noisy_position_sequence, n_particles_per_example, global_context, particle_types
        )
        predicted_normalized_acceleration = self._graph_network(input_graphs_tuple)

        # Adjust next_position by the noise in the last input position
        next_position_adjusted = next_position + position_sequence_noise[:, -1]

        target_normalized_acceleration = self._inverse_decoder_postprocessor(
            next_position_adjusted, noisy_position_sequence
        )

        return predicted_normalized_acceleration, target_normalized_acceleration

    def _inverse_decoder_postprocessor(self, next_position: Tensor, position_sequence: Tensor) -> Tensor:
        """
        Inverse operation of _decoder_postprocessor to get normalized acceleration from known next position.

        Args:
          next_position: [num_particles, num_dimensions]
          position_sequence: [num_particles, seq_length, num_dimensions]

        Returns:
          normalized_acceleration: [num_particles, num_dimensions]
        """
        previous_position = position_sequence[:, -1]
        previous_velocity = previous_position - position_sequence[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity

        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration = (acceleration - acceleration_stats["mean"]) / acceleration_stats["std"]
        return normalized_acceleration
