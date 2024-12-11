"""Training script for Learning to Simulate Complex Physics paper implementation."""

import argparse
import os
import json
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Dataset#, DataLoader
from torch import nn, optim
from typing import Dict, Optional, Tuple, Union, Any

from learned_simulator import LearnedSimulator
from noise_utils import get_random_walk_noise_for_position_sequence
from reading_utils import load_data_from_npz, split_trajectory

INPUT_SEQUENCE_LENGTH = 6  # For calculating last 5 velocities
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

def get_kinematic_mask(particle_types: torch.Tensor) -> torch.Tensor:
    """Returns boolean mask, True for kinematic (obstacle) particles."""
    return (particle_types == KINEMATIC_PARTICLE_ID)

def _read_metadata(data_path: str) -> Dict[str, Any]:
    """Reads and returns the metadata dictionary."""
    with open(os.path.join(data_path, 'metadata.json'), 'r') as fp:
        return json.load(fp)

class Stats:
    """Container for normalization statistics."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

def _combine_std(std_x: Union[float, torch.Tensor], 
                std_y: Union[float, torch.Tensor]) -> torch.Tensor:
    """Combines two independent standard deviations."""
    return torch.sqrt(std_x**2 + std_y**2)

def _get_device() -> torch.device:
    """Returns the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

DEVICE = _get_device()
print(f"Using device: {DEVICE}")

def _get_simulator(
    metadata: Dict[str, Any],
    acc_noise_std: float,
    vel_noise_std: float,
    latent_size: int = 128,
    hidden_size: int = 128,
    hidden_layers: int = 2,
    message_passing_steps: int = 10
) -> LearnedSimulator:
    """Creates and returns the simulator with proper initialization."""
    def cast(v):
        return torch.tensor(v, dtype=torch.float32, device=DEVICE)

    acc_mean = cast(metadata['acc_mean'])
    acc_std = _combine_std(cast(metadata['acc_std']), acc_noise_std)
    acc_stats = Stats(acc_mean, acc_std)

    vel_mean = cast(metadata['vel_mean'])
    vel_std = _combine_std(cast(metadata['vel_std']), vel_noise_std)
    vel_stats = Stats(vel_mean, vel_std)

    normalization_stats = {
        'acceleration': {'mean': acc_stats.mean, 'std': acc_stats.std},
        'velocity': {'mean': vel_stats.mean, 'std': vel_stats.std}
    }

    if 'context_mean' in metadata:
        context_mean = cast(metadata['context_mean'])
        context_std = cast(metadata['context_std'])
        normalization_stats['context'] = {
            'mean': context_mean,
            'std': context_std
        }

    simulator = LearnedSimulator(
        num_dimensions=metadata['dim'],
        connectivity_radius=metadata['default_connectivity_radius'],
        graph_network_kwargs={
            'latent_size': latent_size,
            'mlp_hidden_size': hidden_size,
            'mlp_num_hidden_layers': hidden_layers,
            'num_message_passing_steps': message_passing_steps
        },
        boundaries=metadata['bounds'],
        num_particle_types=NUM_PARTICLE_TYPES,
        normalization_stats=normalization_stats,
        particle_type_embedding_size=16
    ).to(DEVICE)

    return simulator

class OneStepDataset(Dataset):
    """Dataset for training and evaluating single-step predictions."""
    
    def __init__(self, data_path: str, split: str, mode: str = 'one_step_train'):
        """Initializes dataset for one-step predictions.
        
        Args:
            data_path: Path to dataset directory
            split: Which split to load ('train', 'valid', 'test')
            mode: Either 'one_step_train' or 'one_step'
        """
        self.metadata = _read_metadata(data_path)
        self.mode = mode
        self.examples = []
        
        trajectories = load_data_from_npz(data_path, split)
        window_length = INPUT_SEQUENCE_LENGTH + 1
        
        for traj in trajectories:
            windows = split_trajectory(traj, window_length=window_length)
            for w in windows:
                # Create position sequence [num_particles, window_length, dim]
                pos = torch.tensor(w['position'], dtype=torch.float32, device=DEVICE)
                pos = pos.permute(1, 0, 2)
                
                # Split into input and target
                target_position = pos[:, -1]
                input_pos = pos[:, :-1]
                
                particle_type = torch.tensor(w['particle_type'], dtype=torch.int64, device=DEVICE)
                
                # Create node features by flattening position sequence
                n_particles = input_pos.size(0)
                dim = input_pos.size(-1)
                x = input_pos.reshape(n_particles, -1)
                
                # Create PyG Data object
                data = Data(x=x, y=target_position)
                data.particle_type = particle_type
                
                # Add step context if available
                if 'step_context' in w:
                    sc = w['step_context'][window_length-2]
                    data.step_context = torch.tensor(sc, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                else:
                    data.step_context = None
                    
                self.examples.append(data)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class RolloutDataset(Dataset):
    """Dataset for rollout evaluation with preprocessing aligned to training mode."""
    
    def __init__(self, data_path: str, split: str):
        """
        Args:
            data_path: Path to dataset directory
            split: Which split to use ('train', 'valid', 'test')
        """
        self.metadata = _read_metadata(data_path)
        self.trajectories = load_data_from_npz(data_path, split)
        self.sequence_length = self.metadata['sequence_length']
        self.dim = self.metadata['dim']

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        """Returns a trajectory formatted to match training preprocessing."""
        trajectory = self.trajectories[idx]
        
        # Load full position sequence
        position = torch.tensor(
            trajectory['position'],
            dtype=torch.float32,
            device=DEVICE
        )  # [sequence_length, num_particles, dim]
        
        # Get particle information
        num_particles = position.size(1)
        
        # Create input sequence exactly like training mode
        # Take first INPUT_SEQUENCE_LENGTH positions
        input_positions = position[:INPUT_SEQUENCE_LENGTH]  # [6, num_particles, dim]
        input_positions = input_positions.transpose(0, 1)  # [num_particles, 6, dim]
        
        # Flatten the input sequence like in training mode
        x = input_positions.reshape(num_particles, -1)  # [num_particles, 6*dim]
        
        particle_type = torch.tensor(
            trajectory['particle_type'],
            dtype=torch.int64,
            device=DEVICE
        )  # [num_particles]
        
        # Create PyG Data object with same structure as training
        data = Data(
            x=x,  # Flattened input sequence
            position=position,  # Keep full sequence for rollout
            particle_type=particle_type,
            n_particles=torch.tensor([num_particles], device=DEVICE)
        )
        
        # Handle step context in same way as training
        if 'step_context' in trajectory:
            step_context = torch.tensor(
                trajectory['step_context'],
                dtype=torch.float32,
                device=DEVICE
            )  # [sequence_length, context_feat_dim]
            
            # In training, we take the context from specific timesteps
            # Store full context but process it during rollout
            data.step_context = step_context
            
        return data

    def get_input_sequence_shape(self):
        """Helper method to get shape of flattened input sequence."""
        return (INPUT_SEQUENCE_LENGTH * self.dim)

def rollout(
    simulator: LearnedSimulator,
    features: Data,
    num_steps: int,
    metadata: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """Performs rollout of the model."""
    with torch.no_grad():
        
        initial_positions = features.position[:INPUT_SEQUENCE_LENGTH + 1]
        initial_positions = initial_positions.transpose(0, 1)
        ground_truth_positions = features.position[INPUT_SEQUENCE_LENGTH:]
        
        global_context = getattr(features, 'step_context', None)
        current_positions = initial_positions.clone()
        predictions = []
        
        for step in range(num_steps):
            if global_context is not None:
                context_idx = step + INPUT_SEQUENCE_LENGTH - 1
                step_context = global_context[context_idx:context_idx+1]
            else:
                step_context = None
                
            next_position = simulator(
                position_sequence=current_positions,
                n_particles_per_example=features.n_particles,
                particle_types=features.particle_type,
                global_context=step_context
            )
            
            kinematic_mask = get_kinematic_mask(features.particle_type)  # [num_particles]
            next_position_ground_truth = ground_truth_positions[step]  # [num_particles, dim]
            
            # First unsqueeze to add the dim dimension, then broadcast
            kinematic_mask = kinematic_mask.view(-1, 1).expand(-1, metadata['dim'])
            
            next_position = torch.where(
                kinematic_mask,
                next_position_ground_truth,
                next_position
            )
            
            predictions.append(next_position)
            
            current_positions = torch.cat([
                current_positions[:, 1:],
                next_position.unsqueeze(1)
            ], dim=1)
            
        predictions = torch.stack(predictions, dim=0)
        
        return {
            'initial_positions': initial_positions.transpose(0, 1).cpu().numpy(),
            'predicted_rollout': predictions.cpu().numpy(),
            'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
            'particle_types': features.particle_type.cpu().numpy(),
            'metadata': metadata
        }

def train_one_step_model(args):
    """Trains the model for single-step predictions."""
    metadata = _read_metadata(args.data_path)
    simulator = _get_simulator(
        metadata,
        acc_noise_std=args.noise_std,
        vel_noise_std=args.noise_std
    )

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path, exist_ok=True)

    train_dataset = OneStepDataset(args.data_path, 'train', mode='one_step_train')
    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize simulator dimensions with sample batch
    data_iter = iter(train_loader)
    initial_data = next(data_iter).to(DEVICE)

    sequence_length = INPUT_SEQUENCE_LENGTH + 1
    dim = metadata['dim']
    num_nodes = initial_data.x.size(0)

    # Reconstruct position sequence
    input_pos = initial_data.x.view(num_nodes, sequence_length - 1, dim)
    target_position = initial_data.y
    position_sequence = torch.cat([input_pos, target_position.unsqueeze(1)], dim=1)

    # Get batch information
    unique_graph_ids, counts = torch.unique(initial_data.batch, return_counts=True)
    n_particles_per_example = counts

    # Initialize encoder
    example_graph = simulator._encoder_preprocessor(
        position_sequence,
        n_particles_per_example,
        getattr(initial_data, 'step_context', None),
        initial_data.particle_type
    )

    node_in_dim = example_graph.x.size(-1)
    edge_in_dim = example_graph.edge_attr.size(-1) if example_graph.edge_attr is not None else 0
    global_in_dim = example_graph.globals.size(-1) if hasattr(example_graph, 'globals') and example_graph.globals is not None else 0

    simulator._graph_network.initialize_encoder(node_in_dim, edge_in_dim, global_in_dim)
    simulator.to(DEVICE)

    # Reset train loader
    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop
    optimizer = optim.Adam(simulator.parameters(), lr=1e-4)
    step = 0
    simulator.train()

    for epoch in range(1, 2):
        for data in train_loader:
            data = data.to(DEVICE)
            num_nodes = data.x.size(0)

            # Reconstruct position sequence
            input_pos = data.x.view(num_nodes, sequence_length - 1, dim)
            target_position = data.y
            position_sequence = torch.cat([input_pos, target_position.unsqueeze(1)], dim=1)

            # Get batch information and context
            unique_graph_ids, counts = torch.unique(data.batch, return_counts=True)
            n_particles_per_example = counts
            global_context = getattr(data, 'step_context', None)
            particle_types = data.particle_type

            # Add noise
            sampled_noise = get_random_walk_noise_for_position_sequence(
                position_sequence,
                noise_std_last_step=args.noise_std
            ).to(DEVICE)

            # Mask kinematic particles
            non_kin_mask = ~get_kinematic_mask(particle_types)
            noise_mask = non_kin_mask.unsqueeze(-1).unsqueeze(-1).float()
            sampled_noise = sampled_noise * noise_mask

            # Forward pass
            pred_acc, target_acc = simulator.get_predicted_and_target_normalized_accelerations(
                next_position=target_position,
                position_sequence=position_sequence,
                position_sequence_noise=sampled_noise,
                n_particles_per_example=n_particles_per_example,
                particle_types=particle_types,
                global_context=global_context
            )

            # Compute loss (only on non-kinematic particles)
            loss = (pred_acc - target_acc)**2
            loss = loss.sum(dim=-1)
            loss = loss * non_kin_mask.float()
            loss = loss.sum() / non_kin_mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            # Continuing from the training loop...
            if step % 10 == 0:
                print(f"Step {step}, loss {loss.item()}")
                torch.save(simulator.state_dict(), os.path.join(args.model_path, 'checkpoint.pt'))

            if step >= args.num_steps:
                torch.save(simulator.state_dict(), os.path.join(args.model_path, 'checkpoint.pt'))
                print(f"Reached max steps ({args.num_steps}). Final checkpoint saved.")
                return
    
    # Save final checkpoint
    torch.save(simulator.state_dict(), os.path.join(args.model_path, 'checkpoint.pt'))
    print(f"Training completed. Final checkpoint saved at step {step}.")

def eval_one_step_model(args):
    """Evaluates the model on single-step predictions."""
    metadata = _read_metadata(args.data_path)
    simulator = _get_simulator(
        metadata,
        acc_noise_std=args.noise_std,
        vel_noise_std=args.noise_std
    )

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path {args.model_path} does not exist.")

    eval_dataset = OneStepDataset(args.data_path, args.eval_split, mode='one_step')
    eval_loader = PyGDataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize dimensions with sample batch
    data_iter = iter(eval_loader)
    try:
        initial_data = next(data_iter)
    except StopIteration:
        raise ValueError("Evaluation dataset is empty.")
    initial_data = initial_data.to(DEVICE)

    sequence_length = INPUT_SEQUENCE_LENGTH + 1
    dim = metadata['dim']
    num_nodes = initial_data.x.size(0)

    # Reconstruct position sequence
    input_pos = initial_data.x.view(num_nodes, sequence_length - 1, dim)
    target_position = initial_data.y
    position_sequence = torch.cat([input_pos, target_position.unsqueeze(1)], dim=1)

    # Get batch information
    unique_graph_ids, counts = torch.unique(initial_data.batch, return_counts=True)
    n_particles_per_example = counts

    # Initialize encoder
    example_graph = simulator._encoder_preprocessor(
        position_sequence,
        n_particles_per_example,
        getattr(initial_data, 'step_context', None),
        initial_data.particle_type
    )

    node_in_dim = example_graph.x.size(-1)
    edge_in_dim = example_graph.edge_attr.size(-1) if example_graph.edge_attr is not None else 0
    global_in_dim = example_graph.globals.size(-1) if hasattr(example_graph, 'globals') and example_graph.globals is not None else 0

    simulator._graph_network.initialize_encoder(node_in_dim, edge_in_dim, global_in_dim)
    simulator.to(DEVICE)

    # Reset eval loader
    eval_loader = PyGDataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # Load checkpoint
    checkpoint_path = os.path.join(args.model_path, 'checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    if torch.__version__ >= "2.0.0":
        simulator.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    else:
        simulator.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    simulator.eval()

    total_loss = 0
    count = 0
    with torch.no_grad():
        for data in eval_loader:
            data = data.to(DEVICE)
            num_nodes = data.x.size(0)
            
            # Reconstruct position sequence
            input_pos = data.x.view(num_nodes, sequence_length - 1, dim)
            target_position = data.y
            position_sequence = torch.cat([input_pos, target_position.unsqueeze(1)], dim=1)

            # Get batch information
            unique_graph_ids, counts = torch.unique(data.batch, return_counts=True)
            n_particles_per_example = counts
            global_context = getattr(data, 'step_context', None)
            particle_types = data.particle_type

            # Add noise
            sampled_noise = get_random_walk_noise_for_position_sequence(
                position_sequence,
                noise_std_last_step=args.noise_std
            ).to(DEVICE)

            # Mask kinematic particles
            non_kin_mask = ~get_kinematic_mask(particle_types)
            noise_mask = non_kin_mask.unsqueeze(-1).unsqueeze(-1).float()
            sampled_noise = sampled_noise * noise_mask

            # Forward pass
            pred_acc, target_acc = simulator.get_predicted_and_target_normalized_accelerations(
                next_position=target_position,
                position_sequence=position_sequence,
                position_sequence_noise=sampled_noise,
                n_particles_per_example=n_particles_per_example,
                particle_types=particle_types,
                global_context=global_context
            )

            # Compute loss
            loss = (pred_acc - target_acc) ** 2
            loss = loss.sum(dim=-1)
            loss = loss * non_kin_mask.float()
            loss = loss.sum() / non_kin_mask.sum()

            total_loss += loss.item()
            count += 1

    print(f"Eval {args.eval_split} loss: {total_loss / count}")

def rollout_eval(args):
    """Evaluates the model by rolling out entire trajectories."""
    metadata = _read_metadata(args.data_path)
    simulator = _get_simulator(
        metadata,
        acc_noise_std=args.noise_std,
        vel_noise_std=args.noise_std
    )

    # Setup rollout dataset and loader
    rollout_dataset = RolloutDataset(args.data_path, args.eval_split)
    rollout_loader = PyGDataLoader(rollout_dataset, batch_size=1, shuffle=False)

    # Initialize model architecture using first batch
    data_iter = iter(rollout_loader)
    try:
        initial_data = next(data_iter)
        initial_data = initial_data.to(DEVICE)
    except StopIteration:
        raise ValueError("Rollout dataset is empty.")

    num_particles = initial_data.x.size(0)
    dim = metadata['dim']

    # Reshape the sequence
    input_sequence = initial_data.x.view(1, num_particles, INPUT_SEQUENCE_LENGTH, dim)  # Add batch dimension

    # Get and reshape target position
    target_position = initial_data.position[INPUT_SEQUENCE_LENGTH]  # [num_particles, dim]
    target_position = target_position.unsqueeze(0)  # Add batch dimension [1, num_particles, dim]

    # Concatenate along the sequence dimension
    position_sequence = torch.cat([
        input_sequence,
        target_position.unsqueeze(2)  # Add sequence dimension [1, num_particles, 1, dim]
    ], dim=2)  # Concatenate on sequence dimension

    # Remove batch dimension for encoder
    position_sequence = position_sequence.squeeze(0)

    # Initialize encoder
    example_graph = simulator._encoder_preprocessor(
        position_sequence,
        initial_data.n_particles,
        getattr(initial_data, 'step_context', None),
        initial_data.particle_type
    )

    # Get dimensions for initialization
    node_in_dim = example_graph.x.size(-1)
    edge_in_dim = example_graph.edge_attr.size(-1) if example_graph.edge_attr is not None else 0
    global_in_dim = example_graph.globals.size(-1) if hasattr(example_graph, 'globals') and example_graph.globals is not None else 0

    # Initialize encoder with determined dimensions
    simulator._graph_network.initialize_encoder(node_in_dim, edge_in_dim, global_in_dim)
    simulator.to(DEVICE)

    # Reset loader
    rollout_loader = PyGDataLoader(rollout_dataset, batch_size=1, shuffle=False)

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Load checkpoint
    checkpoint_path = os.path.join(args.model_path, 'checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    if torch.__version__ >= "2.0.0":
        simulator.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    else:
        simulator.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    simulator.eval()

    # Perform rollouts
    num_steps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
    
    with torch.no_grad():
        for i, data in enumerate(rollout_loader):
            # move to device
            data = data.to(DEVICE)
            
            # construct input sequence same way as training
            num_particles = data.x.size(0)
            position_sequence = data.x.view(num_particles, INPUT_SEQUENCE_LENGTH, dim)
            
            # Perform rollout
            rollout_results = rollout(
                simulator=simulator,
                features=data,
                num_steps=num_steps,
                metadata=metadata
            )
            
            # Save results
            filename = os.path.join(args.output_path, f'rollout_{args.eval_split}_{i}.pkl')
            print(f'Saving: {filename}')
            with open(filename, 'wb') as f:
                pickle.dump(rollout_results, f)

def main():
    """Main function for training and evaluation."""
    parser = argparse.ArgumentParser(description='Train or evaluate the simulation model.')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'eval_rollout'],
                       help='Train model, one step evaluation or rollout evaluation.')
    parser.add_argument('--eval_split', type=str, default='test',
                       choices=['train', 'valid', 'test'],
                       help='Split to use when running evaluation.')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the dataset directory.')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='The batch size.')
    parser.add_argument('--num_steps', type=int, default=int(2e7),
                       help='Number of training steps.')
    parser.add_argument('--noise_std', type=float, default=6.7e-4,
                       help='The standard deviation of the noise.')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path for saving model checkpoints.')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path for saving rollout outputs.')

    args = parser.parse_args()

    if args.mode == 'train':
        train_one_step_model(args)
    elif args.mode == 'eval':
        eval_one_step_model(args)
    elif args.mode == 'eval_rollout':
        if args.output_path is None:
            raise ValueError('An output path must be provided for eval_rollout.')
        rollout_eval(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    main()