# training.py

import argparse
import os
import json
import pickle
import math
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

from learned_simulator import LearnedSimulator
from noise_utils import get_random_walk_noise_for_position_sequence
from reading_utils import load_data_from_npz, split_trajectory

INPUT_SEQUENCE_LENGTH = 6
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

def get_kinematic_mask(particle_types: torch.Tensor) -> torch.Tensor:
    return (particle_types == KINEMATIC_PARTICLE_ID)

def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'r') as fp:
        return json.load(fp)

class Stats:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

def _combine_std(std_x, std_y):
    return torch.sqrt(std_x**2 + std_y**2)

def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

DEVICE = _get_device()
print(f"Using device: {DEVICE}")

def _get_simulator(
    metadata, 
    acc_noise_std, 
    vel_noise_std, 
    latent_size=128,
    hidden_size=128, 
    hidden_layers=2, 
    message_passing_steps=10
    ):
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
    def __init__(self, data_path, split, mode='one_step_train'):
        self.metadata = _read_metadata(data_path)
        self.mode = mode
        self.examples = []
        trajectories = load_data_from_npz(data_path, split)
        window_length = INPUT_SEQUENCE_LENGTH + 1  # 7

        for traj in trajectories:
            windows = split_trajectory(traj, window_length=window_length)
            for w in windows:
                pos = torch.tensor(w['position'], dtype=torch.float32, device=DEVICE) # [window_length, num_particles, dim]
                pos = pos.permute(1,0,2) # [num_particles, window_length, dim]
                target_position = pos[:, -1]
                input_pos = pos[:, :-1]

                particle_type = torch.tensor(w['particle_type'], dtype=torch.int64, device=DEVICE)
                n_particles = input_pos.size(0)
                seq_length_minus_one = INPUT_SEQUENCE_LENGTH
                dim = input_pos.size(-1)

                # Flatten input_pos into node features:
                # [num_particles, (window_length-1)*dim]
                x = input_pos.reshape(n_particles, seq_length_minus_one * dim)
                y = target_position  # [num_particles, dim]

                data = Data(x=x, y=y)
                data.particle_type = particle_type

                if 'step_context' in w:
                    # w['step_context']: [window_length, context_feat_dim]
                    # The model expects the penultimate step context (like original code)?
                    # In original code, we took step_context at window_length-2 for target.
                    # If you still need that logic, do:
                    sc = w['step_context'][window_length-2]  # shape [context_feat_dim]
                    data.step_context = torch.tensor(sc, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                else:
                    data.step_context = None

                # We now have a Data object representing one example window.
                self.examples.append(data)


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def train_one_step_model(args):
    metadata = _read_metadata(args.data_path)
    simulator = _get_simulator(
        metadata, 
        acc_noise_std=args.noise_std, 
        vel_noise_std=args.noise_std
    )

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path, exist_ok=True)

    train_dataset = OneStepDataset(args.data_path, 'train', mode='one_step_train')

    # Use PyG DataLoader
    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # -------------------------------------------------------------------------
    # Determine correct input dimensions by running a single batch through 
    # the simulator's _encoder_preprocessor.
    # -------------------------------------------------------------------------
    data_iter = iter(train_loader)
    initial_data = next(data_iter)
    initial_data = initial_data.to(DEVICE)

    # We know:
    # initial_data.x: [total_nodes_in_batch, (sequence_length-1)*dim]
    # initial_data.y: [total_nodes_in_batch, dim]

    sequence_length = INPUT_SEQUENCE_LENGTH + 1
    dim = metadata['dim']
    num_nodes = initial_data.x.size(0)

    # Reshape to get position_sequence
    input_pos = initial_data.x.view(num_nodes, sequence_length - 1, dim)   # [num_nodes, sequence_length-1, dim]
    target_position = initial_data.y                                      # [num_nodes, dim]
    position_sequence = torch.cat([input_pos, target_position.unsqueeze(1)], dim=1)  # [num_nodes, sequence_length, dim]

    # Compute n_particles_per_example
    unique_graph_ids, counts = torch.unique(initial_data.batch, return_counts=True)
    n_particles_per_example = counts  # [batch_size]

    # Get global_context if available
    global_context = getattr(initial_data, 'step_context', None)
    particle_types = initial_data.particle_type

    # The encoder_preprocessor will add all necessary features (velocity, boundary distances, particle embeddings, etc.)
    example_graph = simulator._encoder_preprocessor(
        position_sequence,
        n_particles_per_example,
        global_context,
        particle_types
    )

    # Now we can determine the final dimensions:
    node_in_dim = example_graph.x.size(-1)
    edge_in_dim = example_graph.edge_attr.size(-1) if example_graph.edge_attr is not None else 0
    global_in_dim = example_graph.globals.size(-1) if hasattr(example_graph, 'globals') and example_graph.globals is not None else 0
        
    # Initialize the graph network encoder with the correct dimensions
    simulator._graph_network.initialize_encoder(node_in_dim, edge_in_dim, global_in_dim)
    simulator.to(DEVICE)

    # After dimension inference, recreate the train_loader iterator to start from the beginning:
    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(simulator.parameters(), lr=1e-4)
    step = 0
    simulator.train()

    for epoch in range(1, 2):  # Adjust epochs as needed
        for data in train_loader:
            data = data.to(DEVICE)

            num_nodes = data.x.size(0)
            # Reconstruct position_sequence from data.x and data.y
            input_pos = data.x.view(num_nodes, sequence_length - 1, dim)
            target_position = data.y
            position_sequence = torch.cat([input_pos, target_position.unsqueeze(1)], dim=1)

            unique_graph_ids, counts = torch.unique(data.batch, return_counts=True)
            n_particles_per_example = counts
            global_context = getattr(data, 'step_context', None)
            particle_types = data.particle_type

            # Add noise
            sampled_noise = get_random_walk_noise_for_position_sequence(position_sequence, noise_std_last_step=args.noise_std).to(DEVICE)

            # Mask out kinematic particles
            non_kin_mask = ~(get_kinematic_mask(particle_types))
            noise_mask = non_kin_mask.unsqueeze(-1).unsqueeze(-1).float()
            sampled_noise = sampled_noise * noise_mask

            # Forward pass to get predicted and target normalized accelerations
            pred_acc, target_acc = simulator.get_predicted_and_target_normalized_accelerations(
                next_position=target_position,
                position_sequence=position_sequence,
                position_sequence_noise=sampled_noise,
                n_particles_per_example=n_particles_per_example,
                particle_types=particle_types,
                global_context=global_context
            )

            # Compute MSE loss only over non-kinematic particles
            loss = (pred_acc - target_acc)**2
            loss = loss.sum(dim=-1)
            loss = loss * non_kin_mask.float()
            loss = loss.sum() / non_kin_mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 500 == 0:
                print(f"Step {step}, loss {loss.item()}")
                torch.save(simulator.state_dict(), os.path.join(args.model_path, 'checkpoint.pt'))

            if step >= args.num_steps:
                torch.save(simulator.state_dict(), os.path.join(args.model_path, 'checkpoint.pt'))
                print(f"Reached max steps ({args.num_steps}). Final checkpoint saved.")
                return
    
    # Save a final checkpoint after training completes
    torch.save(simulator.state_dict(), os.path.join(args.model_path, 'checkpoint.pt'))
    print(f"Training completed. Final checkpoint saved at step {step}.")

def eval_one_step_model(args):
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

    # init model's architecture using a sample batch --> differs from TF code
    data_iter = iter(eval_loader)
    try:
        initial_data = next(data_iter)
    except StopIteration:
        raise ValueError("Evaluation dataset is empty.")
    initial_data = initial_data.to(DEVICE)

    sequence_length = INPUT_SEQUENCE_LENGTH + 1
    dim = metadata['dim']
    num_nodes = initial_data.x.size(0)

    # reshape to get position_sequence
    input_pos = initial_data.x.view(num_nodes, sequence_length - 1, dim)   # [num_particles, sequence_length-1, dim]
    target_position = initial_data.y                                      # [num_particles, dim]
    position_sequence = torch.cat([input_pos, target_position.unsqueeze(1)], dim=1)  # [num_particles, sequence_length, dim]

    # compute n_particles_per_example
    unique_graph_ids, counts = torch.unique(initial_data.batch, return_counts=True)
    n_particles_per_example = counts  # [batch_size]

    global_context = getattr(initial_data, 'step_context', None)
    particle_types = initial_data.particle_type

    # pass thro encoder preprocessor to init encoder
    example_graph = simulator._encoder_preprocessor(
        position_sequence,
        n_particles_per_example,
        global_context,
        particle_types
    )

    # determine the dims
    node_in_dim = example_graph.x.size(-1)
    edge_in_dim = example_graph.edge_attr.size(-1) if example_graph.edge_attr is not None else 0
    global_in_dim = example_graph.globals.size(-1) if hasattr(example_graph, 'globals') and example_graph.globals is not None else 0

    # init encoder with determined dims
    simulator._graph_network.initialize_encoder(node_in_dim, edge_in_dim, global_in_dim)
    simulator.to(DEVICE)

    # reload the eval_loader iterator to start from the beginning
    eval_loader = PyGDataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # load state_dict
    checkpoint_path = os.path.join(args.model_path, 'checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    #? INCLUDE PYTORCH VERSION IN ENV.YML
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
            input_pos = data.x.view(num_nodes, sequence_length - 1, dim)
            target_position = data.y
            position_sequence = torch.cat([input_pos, target_position.unsqueeze(1)], dim=1)

            unique_graph_ids, counts = torch.unique(data.batch, return_counts=True)
            n_particles_per_example = counts
            global_context = getattr(data, 'step_context', None)
            particle_types = data.particle_type

            # add noise
            sampled_noise = get_random_walk_noise_for_position_sequence(position_sequence, noise_std_last_step=args.noise_std).to(DEVICE)

            # mask out kinematic particles
            non_kin_mask = ~(get_kinematic_mask(particle_types))
            noise_mask = non_kin_mask.unsqueeze(-1).unsqueeze(-1).float()
            sampled_noise = sampled_noise * noise_mask

            # forward pass to get predicted and target normalized accelerations
            pred_acc, target_acc = simulator.get_predicted_and_target_normalized_accelerations(
                next_position=target_position,
                position_sequence=position_sequence,
                position_sequence_noise=sampled_noise,
                n_particles_per_example=n_particles_per_example,
                particle_types=particle_types,
                global_context=global_context
            )

            # MSE loss ONLY over NON-kinematic particles
            loss = (pred_acc - target_acc) ** 2
            loss = loss.sum(dim=-1)
            loss = loss * non_kin_mask.float()
            loss = loss.sum() / non_kin_mask.sum()

            total_loss += loss.item()
            count += 1

    print(f"Eval {args.eval_split} loss: {total_loss / count}")

from torch_geometric.data import Data

# TO DO: BROKEN / UNECESSARY
class RolloutDataset(Dataset):
    """
    A PyTorch Dataset class for handling trajectory data during rollout evaluations.

    Each item in the dataset represents a complete trajectory, which includes:
    - Position sequences for all particles over time.
    - Particle types.
    - Optional step contexts.

    Args:
        data_path (str): Path to the dataset directory containing .npz files.
        split (str): The dataset split to use ('train', 'valid', 'test').
    """
    def __init__(self, data_path, split):
        self.metadata = _read_metadata(data_path)
        self.trajectories = load_data_from_npz(data_path, split)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        pos = torch.tensor(traj['position'], dtype=torch.float32)  # [sequence_length, num_particles, dim]
        particle_type = torch.tensor(traj['particle_type'], dtype=torch.int64)  # [num_particles]
        step_context = None
        if 'step_context' in traj and traj['step_context'] is not None:
            step_context = torch.tensor(traj['step_context'], dtype=torch.float32)  # [sequence_length, context_feat_dim]

        # simulate OneStepDataset structure
        sequence_length = pos.size(0)
        dim = self.metadata['dim']
        num_particles = pos.size(1)

        # Create data.x and data.y similar to OneStepDataset
        input_pos = pos[:INPUT_SEQUENCE_LENGTH].reshape(num_particles, -1)  # [num_particles, 6*dim]
        target_position = pos[INPUT_SEQUENCE_LENGTH]  # [num_particles, dim]

        data = Data()
        data.x = input_pos  # [num_particles, 6*dim]
        data.y = target_position  # [num_particles, dim]
        data.particle_type = particle_type
        if step_context is not None:
            # Assuming step_context at the penultimate step
            sc = step_context[INPUT_SEQUENCE_LENGTH - 1]  # [context_feat_dim]
            data.step_context = sc.unsqueeze(0)  # [1, context_feat_dim]
        else:
            data.step_context = None
        return data

# TO DO: BROKEN / UNECESSARY
def initialize_simulator(simulator, data, metadata):
    """
    Initializes the simulator's encoder based on a sample batch.

    Args:
        simulator (LearnedSimulator): The simulator model to initialize.
        data (Data): A sample Data object from the DataLoader.
        metadata (dict): Metadata dictionary containing dataset information.

    Returns:
        simulator (LearnedSimulator): The initialized simulator.
    """
    data = data.to(DEVICE)
    pos = data.x.view(-1, INPUT_SEQUENCE_LENGTH * metadata['dim'])  # [num_particles, 6*dim]
    target_position = data.y  # [num_particles, dim]
    position_sequence = torch.cat([pos, target_position], dim=1)  # [num_particles, 6*dim + dim] = [num_particles, 7*dim]

    n_particles_per_example = torch.tensor([pos.size(0)], dtype=torch.int64, device=DEVICE)

    global_context = None
    if hasattr(data, 'step_context') and data.step_context is not None:
        global_context = data.step_context  # [1, context_feat_dim]

    example_graph = simulator._encoder_preprocessor(
        position_sequence,
        n_particles_per_example,
        global_context,
        data.particle_type
    )

    node_in_dim = example_graph.x.size(-1)
    edge_in_dim = example_graph.edge_attr.size(-1) if example_graph.edge_attr is not None else 0
    global_in_dim = example_graph.globals.size(-1) if hasattr(example_graph, 'globals') and example_graph.globals is not None else 0

    #? DEBUG
    print(f"node_in_dim: {node_in_dim}, edge_in_dim: {edge_in_dim}, global_in_dim: {global_in_dim}")

    simulator._graph_network.initialize_encoder(node_in_dim, edge_in_dim, global_in_dim)
    simulator.to(DEVICE)

    return simulator

# TO DO: BROKEN
def rollout_eval(args):
    metadata = _read_metadata(args.data_path)
    simulator = _get_simulator(metadata, acc_noise_std=args.noise_std, vel_noise_std=args.noise_std)
    
    #? WORK IN PROGRESS
    class RolloutDataset(Dataset):
        def __init__(self, data_path, split):
            self.metadata = _read_metadata(data_path)
            self.trajectories = load_data_from_npz(data_path, split)

        def __len__(self):
            return len(self.trajectories)

        def __getitem__(self, idx):
            traj = self.trajectories[idx]
            pos = torch.tensor(traj['position'], dtype=torch.float32)  # [sequence_length, num_particles, dim]
            particle_type = torch.tensor(traj['particle_type'], dtype=torch.int64)  # [num_particles]
            step_context = None
            if 'step_context' in traj and traj['step_context'] is not None:
                step_context = torch.tensor(traj['step_context'], dtype=torch.float32)  # [sequence_length, context_feat_dim]

            # simualte OneStepDataset structure
            sequence_length = pos.size(0)
            dim = self.metadata['dim']
            num_particles = pos.size(1)

            # create data.x and data.y like in OneStepDataset
            input_pos = pos[:INPUT_SEQUENCE_LENGTH].reshape(num_particles, -1)  # [num_particles, 6*dim]
            target_position = pos[INPUT_SEQUENCE_LENGTH]  # [num_particles, dim]

            data = Data()
            data.x = input_pos  # [num_particles, 6*dim]
            data.y = target_position  # [num_particles, dim]
            data.particle_type = particle_type
            if step_context is not None:
                # assumes step_context at the penultimate step
                sc = step_context[INPUT_SEQUENCE_LENGTH - 1]  # [context_feat_dim]
                data.step_context = sc.unsqueeze(0)  # [1, context_feat_dim]
            else:
                data.step_context = None
            return data

    # init RolloutDataset and PyGDataLoader
    rollout_dataset = RolloutDataset(args.data_path, args.eval_split)
    rollout_loader = PyGDataLoader(rollout_dataset, batch_size=1, shuffle=False)

    if len(rollout_dataset) == 0:
        raise ValueError("Rollout dataset is empty. Cannot initialize the simulator.")

    # get sample trajectory to init the encoder
    sample_traj = next(iter(rollout_loader))
    pos = sample_traj.x.squeeze(0)  # [sequence_length, num_particles, dim]
    particle_type = sample_traj.particle_type.squeeze(0)  # [num_particles]
    step_context = None
    if hasattr(sample_traj, 'step_context') and sample_traj.step_context is not None:
        step_context = sample_traj.step_context.squeeze(0)  # [1, context_feat_dim]

    sequence_length = pos.size(0)
    dim = metadata['dim']
    num_particles = pos.size(1)  # [num_particles, sequence_length, dim]

    # reshape to match the expected format for the encoder preprocessor
    input_pos = pos[:INPUT_SEQUENCE_LENGTH].view(num_particles, INPUT_SEQUENCE_LENGTH * dim)  # [num_particles, 6*dim]
    target_position = sample_traj.y  # [num_particles, dim]
    position_sequence = torch.cat([input_pos, target_position], dim=1)  # [num_particles, 6*dim + dim] = [num_particles, 7*dim]

    # compute n_particles_per_example (since batch_size=1, it's a SINGLE count)
    n_particles_per_example = torch.tensor([num_particles], dtype=torch.int64, device=DEVICE)

    global_context = None
    if step_context is not None:
        global_context = step_context  # [1, context_feat_dim]

    # pass thro the encoder preprocessor to init encoder
    example_graph = simulator._encoder_preprocessor(
        position_sequence,
        n_particles_per_example,
        global_context,
        particle_type
    )

    # determine dims
    node_in_dim = example_graph.x.size(-1)
    edge_in_dim = example_graph.edge_attr.size(-1) if example_graph.edge_attr is not None else 0
    global_in_dim = example_graph.globals.size(-1) if hasattr(example_graph, 'globals') and example_graph.globals is not None else 0

    #? DEBUG
    print(f"node_in_dim: {node_in_dim}, edge_in_dim: {edge_in_dim}, global_in_dim: {global_in_dim}")

    # init encoder with the determined dims
    simulator._graph_network.initialize_encoder(node_in_dim, edge_in_dim, global_in_dim)
    simulator.to(DEVICE)

    # load the state_dict
    checkpoint_path = os.path.join(args.model_path, 'checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    #? DEPRECATE / SPECIFY IN ENV.YML
    if torch.__version__ >= "2.0.0":
        simulator.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=True)
    else:
        simulator.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=True)

    simulator.eval()

    os.makedirs(args.output_path, exist_ok=True)

    with torch.no_grad():
        for i, traj_batch in enumerate(rollout_loader):
            traj = traj_batch  # traj_batch is a Data object with batch size 1

            pos = traj.x.squeeze(0)  # [sequence_length, num_particles, dim]
            particle_type = traj.particle_type.squeeze(0)  # [num_particles]
            step_context = None
            if hasattr(traj, 'step_context') and traj.step_context is not None:
                step_context = traj.step_context.squeeze(0)  # [1, context_feat_dim]

            sequence_length = pos.size(0)
            num_steps = sequence_length - INPUT_SEQUENCE_LENGTH
            full_pos = pos.permute(1, 0, 2)  # [num_particles, sequence_length, dim]
            initial_positions = full_pos[:, :INPUT_SEQUENCE_LENGTH]  # [num_particles, 6, dim]
            ground_truth_positions = full_pos[:, INPUT_SEQUENCE_LENGTH:]  # [num_particles, 1, dim]
            current_positions = initial_positions.clone()
            predictions = []

            for step in range(num_steps):
                if step_context is not None:
                    context_index = step + INPUT_SEQUENCE_LENGTH - 1
                    if context_index >= step_context.size(0):
                        raise IndexError(f"Context index {context_index} out of bounds for step_context with size {step_context.size(0)}")
                    global_context_step = step_context  # Assuming single context feature
                else:
                    global_context_step = None

                # Prepare position_sequence for the simulator
                # current_positions: [num_particles, 6, dim]
                position_sequence = current_positions.reshape(num_particles, -1)  # [num_particles, 6*dim]

                # Pass through the simulator to get the next position
                next_position = simulator(
                    position_sequence=position_sequence,
                    n_particles_per_example=torch.tensor([num_particles], dtype=torch.int64, device=DEVICE),
                    particle_types=particle_type,
                    global_context=global_context_step
                )  # [num_particles, dim]

                # Replace kinematic particles with ground truth positions
                kinematic_mask = get_kinematic_mask(particle_type)  # [num_particles]
                next_position_gt = ground_truth_positions[:, step]  # [num_particles, dim]
                next_position = torch.where(kinematic_mask.unsqueeze(-1), next_position_gt, next_position)

                # Append the predicted position
                predictions.append(next_position.unsqueeze(1))  # [num_particles, 1, dim]

                # Update current_positions for the next step by shifting and adding the new position
                current_positions = torch.cat([current_positions[:, 1:, :], next_position.unsqueeze(1)], dim=1)  # [num_particles, 6, dim]

            # Concatenate all predictions along the sequence dimension
            predicted_rollout = torch.cat(predictions, dim=1)  # [num_particles, num_steps, dim]

            # Organize the rollout data
            example_rollout = {
                'initial_positions': initial_positions.transpose(0, 1).cpu().numpy(),  # [6, num_particles, dim]
                'predicted_rollout': predicted_rollout.cpu().numpy(),  # [num_particles, num_steps, dim]
                'ground_truth_rollout': ground_truth_positions.cpu().numpy(),  # [num_particles, num_steps, dim]
                'particle_types': particle_type.cpu().numpy(),
                'metadata': metadata
            }
            if step_context is not None:
                example_rollout['global_context'] = step_context.cpu().numpy()

            # Save the rollout
            filename = os.path.join(args.output_path, f'rollout_{args.eval_split}_{i}.pkl')
            print(f'Saving: {filename}')
            with open(filename, 'wb') as f:
                pickle.dump(example_rollout, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'eval_rollout'])
    parser.add_argument('--eval_split', type=str, default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=int(2e7))
    parser.add_argument('--noise_std', type=float, default=6.7e-4)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()

    if args.mode == 'train':
        train_one_step_model(args)
    elif args.mode == 'eval':
        eval_one_step_model(args)
    elif args.mode == 'eval_rollout':
        if args.output_path is None:
            raise ValueError('A rollout path must be provided for eval_rollout.')
        rollout_eval(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    main()
