# Standard library imports
import os
import math
import pickle
import argparse
from collections import deque
from typing import Optional, Dict, Union, List

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

# Local imports
from learned_simulator import LearnedSimulator
from noise_utils import get_random_walk_noise_for_position_sequence
from dataloader import OneStepDataset, RolloutDataset, one_step_collate
from rollout import rollout
from utils import (
    fix_seed,
    _combine_std,
    _read_metadata,
    get_kinematic_mask,
    print_args,
    Stats,
    NUM_PARTICLE_TYPES,
    INPUT_SEQUENCE_LENGTH,
    device,
    compute_adaptive_radius
)

def _get_simulator(
    model_kwargs: dict,
    metadata: dict,
    acc_noise_std: float,
    vel_noise_std: float,
    args
) -> 'LearnedSimulator':
    """
    Initialize simulator with proper normalization statistics and adaptive sampling.
    """
    # Cast metadata values for normalization
    cast = lambda v: np.array(v, dtype=np.float32)

    acceleration_stats = Stats(
        cast(metadata['acc_mean']),
        _combine_std(cast(metadata['acc_std']), acc_noise_std)
    )
    velocity_stats = Stats(
        cast(metadata['vel_mean']),
        _combine_std(cast(metadata['vel_std']), vel_noise_std)
    )

    normalization_stats = {
        'acceleration': acceleration_stats,
        'velocity': velocity_stats,
    }

    if 'context_mean' in metadata:
        context_stats = Stats(
            cast(metadata['context_mean']),
            cast(metadata['context_std'])
        )
        normalization_stats['context'] = context_stats

    # Pad or truncate positions to ensure consistent dimensions
    max_length = max(pos.shape[1] for pos in metadata['positions'])  # Maximum number of particles
    dim = metadata['positions'][0].shape[2]  # Dimensionality of positions

    padded_positions = []
    for pos in metadata['positions']:
        if pos.shape[1] < max_length:  # Pad
            padding = np.zeros((pos.shape[0], max_length - pos.shape[1], dim))
            padded_positions.append(np.concatenate((pos, padding), axis=1))
        elif pos.shape[1] > max_length:  # Truncate
            padded_positions.append(pos[:, :max_length, :])
        else:  # No padding/truncation needed
            padded_positions.append(pos)

    padded_positions = np.array(padded_positions, dtype=np.float32)  # Convert list to NumPy array
    print(f"Padded positions shape: {padded_positions.shape}")
    padded_positions = torch.tensor(padded_positions, dtype=torch.float32)

    # Check and compute gradients dynamically if missing
    if 'gradients' not in metadata:
        print("Gradients key missing. Computing gradients dynamically.")
        metadata['gradients'] = []
        for example_positions in metadata['positions']:
            gradients = np.linalg.norm(
                np.diff(example_positions, axis=0), axis=-1
            ).mean(axis=0)  # Mean gradient magnitude across time
            metadata['gradients'].append(gradients)

    # Compute adaptive radius for each example
    adaptive_radii = []
    for example_index, example_positions in enumerate(metadata['positions']):
        gradients = torch.tensor(metadata['gradients'][example_index], dtype=torch.float32)
        base_radius = metadata['default_connectivity_radius']
        adaptive_radius = compute_adaptive_radius(
            positions=example_positions,
            gradients=gradients,
            base_radius=base_radius
        )
        adaptive_radii.append(adaptive_radius)

    # Validate and reshape adaptive radii
    all_radii = []
    for radius, pos in zip(adaptive_radii, metadata['positions']):
        num_particles = pos.shape[1]
        if len(radius.shape) == 1 and radius.shape[0] == num_particles:
            all_radii.append(radius)
        else:
            raise ValueError(f"Radius shape mismatch for example: {radius.shape}, expected ({num_particles},)")

    # Flatten all radii for the simulator
    adaptive_radius = torch.cat(all_radii, dim=0)
    print(f"Adaptive radius shape: {adaptive_radius.shape}, expected total particles: {sum(pos.shape[1] for pos in metadata['positions'])}")
    assert adaptive_radius.shape[0] == sum(pos.shape[1] for pos in metadata['positions']), (
        f"Radius shape mismatch: expected ({sum(pos.shape[1] for pos in metadata['positions'])},), got {adaptive_radius.shape}"
    )

    # Initialize the simulator with adaptive sampling
    simulator = LearnedSimulator(
        num_dimensions=metadata['dim'],
        connectivity_radius=adaptive_radius,
        graph_network_kwargs=model_kwargs,
        boundaries=metadata['bounds'],
        num_particle_types=NUM_PARTICLE_TYPES,
        normalization_stats=normalization_stats,
        device=device,
        particle_type_embedding_size=16,
        args=args,
    )
    return simulator




def eval_one_step(args: argparse.Namespace) -> None:
    """Evaluate model on single-step predictions.
    
    Loads a trained model and evaluates its performance on single-step predictions
    using the specified dataset split. Calculates MSE loss for both position and
    acceleration predictions.
    
    Args:
        args: Namespace containing:
            - dataset: Name of the dataset to evaluate on
            - eval_split: Which data split to use (train/valid/test)
            - batch_size: Batch size for evaluation
            - model_path: Path to saved model checkpoints
            - gnn_type: Type of GNN used
            - noise_std: Standard deviation for noise
            - message_passing_steps: Number of message passing steps
            
    Raises:
        ValueError: If no model checkpoint is found
    """
    # Data setup
    sequence_dataset = OneStepDataset(args.dataset, args.eval_split)
    sequence_dataloader = DataLoader(
        sequence_dataset, 
        collate_fn=one_step_collate, 
        batch_size=args.batch_size, 
        shuffle=False
    )

    # Model initialization
    metadata = _read_metadata(data_path=f"datasets/{args.dataset}")
    model_kwargs = dict(
        latent_size=128,
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=args.message_passing_steps,
    )
    
    # Initialize simulator
    simulator = _get_simulator(
        model_kwargs=model_kwargs,
        metadata=metadata,
        vel_noise_std=args.noise_std,
        acc_noise_std=args.noise_std,
        args=args
    )
    
    # Load model checkpoint
    checkpoint_path = f'{args.model_path}/{args.dataset}/{args.gnn_type}'
    checkpoint_file = None
    for file in os.listdir(checkpoint_path):
        if file.startswith('lowest_train_mse'):
            checkpoint_file = os.path.join(checkpoint_path, file)
            break
    
    if not checkpoint_file:
        raise ValueError("No checkpoint exists!")
    print(f"Load checkpoint from: {checkpoint_file}")
    
    simulator_state_dict = torch.load(checkpoint_file, map_location=device)
    simulator.load_state_dict(simulator_state_dict)

    # Evaluation loop
    mse_loss = F.mse_loss
    total_loss = []
    time_step = 0

    print("################### Begin Evaluate One Step #######################")
    with torch.no_grad():
        for features, labels in sequence_dataloader:
            # Move data to device
            labels = labels.to(device)
            target_next_position = labels
            
            # Move features to device
            features['positions'] = features['positions'].to(device)
            features['particle_types'] = features['particle_types'].to(device)
            features['n_particles_per_example'] = features['n_particles_per_example'].to(device)
            if 'step_context' in features:
                features['step_context'] = features['step_context'].to(device)

            # Generate noise and get predictions
            sampled_noise = get_random_walk_noise_for_position_sequence(
                features['positions'],
                noise_std_last_step=args.noise_std
            ).to(device)

            predicted_next_position = simulator(
                position_sequence=features['positions'],
                n_particles_per_example=features['n_particles_per_example'],
                particle_types=features['particle_types'],
                global_context=features.get('step_context')
            )

            pred_target = simulator.get_predicted_and_target_normalized_accelerations(
                next_position=target_next_position,
                position_sequence=features['positions'],
                position_sequence_noise=sampled_noise,
                n_particles_per_example=features['n_particles_per_example'],
                particle_types=features['particle_types'],
                global_context=features.get('step_context')
            )
            pred_acceleration, target_acceleration = pred_target

            # Calculate losses
            loss_mse = mse_loss(pred_acceleration, target_acceleration)
            one_step_position_mse = mse_loss(predicted_next_position, target_next_position)
            total_loss.append(one_step_position_mse)
            
            print(f"step: {time_step}\t loss_mse: {loss_mse:.2f}\t "
                  f"one_step_position_mse: {one_step_position_mse * 1e9:.2f}e-9.")
            time_step += 1

        # Calculate and print average loss
        average_loss = torch.tensor(total_loss).mean().item()
        print(f"Average one step loss is: {average_loss * 1e9}e-9.")

def eval_rollout(args):
    """Evaluate model on trajectory rollouts."""
    # data setup
    sequence_dataset = RolloutDataset(args.dataset, args.eval_split)
    sequence_dataloader = DataLoader(
        sequence_dataset,
        collate_fn=one_step_collate,
        batch_size=1,
        shuffle=False
    )

    metadata = _read_metadata(data_path=f"datasets/{args.dataset}")
    model_kwargs = dict(
        latent_size=128,
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=args.message_passing_steps,
    )

    simulator = _get_simulator(
        model_kwargs,
        metadata,
        vel_noise_std=args.noise_std,
        acc_noise_std=args.noise_std,
        args=args
    )

    num_steps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH

    # load model checkpoint
    model_path = f'{args.model_path}/{args.dataset}/{args.gnn_type}'
    output_path = f'{args.output_path}/{args.dataset}/{args.gnn_type}'
    files = os.listdir(model_path)
    file_name = None

    for file in files:
        if file.startswith('lowest_train_mse'):
            file_name = os.path.join(model_path, file)
            break

    if not file_name:
        raise ValueError("No checkpoint exists!")
    else:
        print(f"Load checkpoint from: {file_name}")

    simulator_state_dict = torch.load(file_name, map_location=device)
    simulator.load_state_dict(simulator_state_dict)

    # evaluation loop
    mse_loss = F.mse_loss
    total_loss = []
    time_step = 0

    print("################### Begin Evaluate Rollout #######################")
    with torch.no_grad():
        for feature, _ in sequence_dataloader:
            feature['positions'] = feature['positions'].to(device)
            feature['particle_types'] = feature['particle_types'].to(device)
            feature['n_particles_per_example'] = feature['n_particles_per_example'].to(device)
            if 'step_context' in feature:
                feature['step_context'] = feature['step_context'].to(device)

            # run rollout
            rollout_op = rollout(simulator, feature, num_steps)
            rollout_op['metadata'] = metadata
            
            # calculate losses
            loss_mse = mse_loss(
                rollout_op['predicted_rollout'], 
                rollout_op['ground_truth_rollout']
            )
            total_loss.append(loss_mse)
            print(f"step: {time_step}\t rollout_loss_mse: {loss_mse * 1e3:.2f}e-3.")

            # save rollout results
            file_name = f'rollout_{args.eval_split}_{time_step}.pkl'
            file_name = os.path.join(output_path, file_name)
            print(f"Saving rollout file {time_step}.")
            with open(file_name, 'wb') as file:
                pickle.dump(rollout_op, file)
            time_step += 1

        average_loss = torch.tensor(total_loss).mean().item()
        print(f"Average rollout loss is: {average_loss * 1e3:.2f}e-3.")

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Creates learning rate scheduler with warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)

def train(args: argparse.Namespace) -> None:
    """Train the simulator model.
    
    Performs training of the particle simulator using the provided configuration.
    Includes periodic validation and model checkpointing of best performing models.
    
    Args:
        args: Configuration namespace containing:
            - seed: Random seed for reproducibility
            - dataset: Name of dataset to train on
            - batch_size: Training batch size
            - noise_std: Standard deviation for noise injection
            - lr: Initial learning rate
            - weight_decay: Weight decay for optimizer
            - num_epochs: Number of training epochs
            - test_step: Steps between validation
            - model_path: Path to save model checkpoints
            - message_passing_steps: Number of message passing steps
            - gnn_type: Type of GNN model used
    """
    fix_seed(args.seed)

    # Data setup
    train_dataset = OneStepDataset(args.dataset, 'train')
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=one_step_collate,
        batch_size=args.batch_size,
        shuffle=True
    )

    # Calculate steps and epochs
    steps_per_epoch = len(train_dataloader) # steps == batches
    total_steps = args.num_epochs * steps_per_epoch
    print(f"\nTraining for {args.num_epochs} epochs with {steps_per_epoch} steps per epoch")
    print(f"Total training steps: {total_steps}")

    # Model initialization
    metadata = _read_metadata(data_path=f"datasets/{args.dataset}")
    model_kwargs = dict(
        latent_size=128,
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=args.message_passing_steps,
        num_heads = 4
    )

    simulator = _get_simulator(
        model_kwargs=model_kwargs,
        metadata=metadata,
        vel_noise_std=args.noise_std,
        acc_noise_std=args.noise_std,
        args=args
    )

    # Optimization setup
    optimizer = torch.optim.AdamW(
        simulator.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler setup
    warmup_steps = min(500, total_steps // 10)  # 10% of total steps or 1000, whichever is smaller
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    mse_loss = F.mse_loss
    max_grad_norm = 1.0
    best_train_loss = float("inf")
    global_step = 0

    # Early stopping setup
    patience = 500  # Number of steps to wait for improvement
    min_delta = 1e-11  # Minimum change in loss to qualify as an improvement
    steps_without_improvement = 0
    
    # Loss smoothing setup
    smoothing_window = 100
    loss_deque = deque(maxlen=smoothing_window)

    # Create fixed checkpoint path
    checkpoint_dir = f'{args.model_path}/{args.dataset}/{args.gnn_type}'
    best_model_path = f'{checkpoint_dir}/lowest_train_mse.pt'

    # Training loop
    print("\Starting training...")
    try:
        for epoch in range(args.num_epochs):
            print(f'Starting Epoch [{epoch+1}/{args.num_epochs}]')
            simulator.train()

            for features, labels in train_dataloader:
                # Move data to device
                labels = labels.to(device)
                target_next_position = labels
                
                for key in ['positions', 'particle_types', 'n_particles_per_example']:
                    features[key] = features[key].to(device)
                if 'step_context' in features:
                    features['step_context'] = features['step_context'].to(device)

                # Add noise with masking
                sampled_noise = get_random_walk_noise_for_position_sequence(
                    features['positions'],
                    noise_std_last_step=args.noise_std
                ).to(device)
                
                non_kinematic_mask = torch.logical_not(get_kinematic_mask(features['particle_types']))
                noise_mask = non_kinematic_mask.unsqueeze(1).unsqueeze(2)
                sampled_noise *= noise_mask

                # Forward pass and loss calculation
                optimizer.zero_grad()
                pred_target = simulator.get_predicted_and_target_normalized_accelerations(
                    next_position=target_next_position,
                    position_sequence=features['positions'],
                    position_sequence_noise=sampled_noise,
                    n_particles_per_example=features['n_particles_per_example'],
                    particle_types=features['particle_types'],
                    global_context=features.get('step_context')
                )
                pred_acceleration, target_acceleration = pred_target

                # Calculate loss
                loss = (pred_acceleration[non_kinematic_mask] - 
                    target_acceleration[non_kinematic_mask]) ** 2
                num_non_kinematic = torch.sum(non_kinematic_mask.to(torch.float32))
                loss = torch.sum(loss) / torch.sum(num_non_kinematic)

                # Optimization step
                loss.mean().backward()
                clip_grad_norm_(simulator.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                predicted_next_position = simulator(
                    position_sequence=features['positions'],
                    n_particles_per_example=features['n_particles_per_example'],
                    particle_types=features['particle_types'],
                    global_context=features.get('step_context')
                )
                position_mse = mse_loss(predicted_next_position, target_next_position)

                # track loss history for convergence check
                loss_deque.append(position_mse.item())
                if len(loss_deque) == smoothing_window:
                    smoothed_loss = sum(loss_deque) / len(loss_deque)

                    # check for improvement
                    if smoothed_loss < best_train_loss - min_delta:
                        best_train_loss = smoothed_loss
                        steps_without_improvement = 0
                        print(f"\nSaving new best model - Step: {global_step}, "
                              f"Training MSE: {smoothed_loss:.4e}\n")
                        torch.save(simulator.state_dict(), best_model_path)
                else:
                    steps_without_improvement += 1
                
                # print progress and update LR every 100 steps
                if global_step % 100 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    smoothed_loss = sum(loss_deque) / len(loss_deque) if loss_deque else position_mse.item()
                    print(f'Step [{global_step}/{total_steps}] '
                          f'Loss: {loss.item():.4f} MSE: {smoothed_loss:.4e}')

                    # Save checkpoints
                    torch.save(
                        simulator.state_dict(),
                        f'{checkpoint_dir}/checkpoint.pt'
                    )
                
                if steps_without_improvement >= patience:
                    print(f"\nEarly stopping triggered - No improvement for {patience} steps")
                    return
                
                global_step += 1

            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch + 1} completed - Current LR: {current_lr:.2e}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        print(f"Saving current checkpoint - Step: {global_step}, "
              f"Training MSE: {position_mse:.4e}")
        torch.save(
            simulator.state_dict(),
            f'{checkpoint_dir}/checkpoint.pt'
        )
    
    print("\nTraining completed!")
    print(f"Best loss achieved: {best_train_loss:.4e}")
    print(f"Best model saved at: {best_model_path}")

def parse_arguments():
    """Parse command line arguments organized by usage mode."""
    parser = argparse.ArgumentParser(description="Learning to Simulate.")
    
    # Global arguments (used by all modes)
    global_group = parser.add_argument_group('Global Arguments')
    global_group.add_argument('--mode', default='train',
                        choices=['train', 'eval', 'eval_rollout'],
                        help='Train model, one step evaluation or rollout evaluation.')
    global_group.add_argument('--dataset', default="Water", type=str,
                        help='The dataset directory.')
    global_group.add_argument('--batch_size', default=2, type=int,
                        help='The batch size for training and evaluation.')
    global_group.add_argument('--model_path', default="models", type=str,
                        help='The path for saving/loading model checkpoints.')
    global_group.add_argument('--gnn_type', default='gcn', 
                        choices=['gcn', 'gat', 'trans_gnn', 'interaction_net'],
                        help='The GNN to be used as processor.')
    global_group.add_argument('--message_passing_steps', default=10, type=int,
                        help='Number of GNN message passing steps.')
    global_group.add_argument('--noise_std', default=0.0003, type=float,
                        help='The std deviation of the noise for training and evaluation.')
    
    # Training-specific arguments
    train_group = parser.add_argument_group('Training Arguments (only used when mode=train)')
    train_group.add_argument('--num_epochs', default=10, type=int,
                        help='Maximum number of training epochs.')
    train_group.add_argument('--seed', type=int, default=483,
                        help='Random seed for reproducibility.')
    train_group.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate.')
    train_group.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay for optimizer.')
    
    # Evaluation-specific arguments
    eval_group = parser.add_argument_group('Evaluation Arguments (only used when mode=eval or eval_rollout)')
    eval_group.add_argument('--eval_split', default='test',
                        choices=['train', 'valid', 'test'],
                        help='Dataset split to use for evaluation.')
    eval_group.add_argument('--output_path', default="rollouts", type=str,
                        help='Path for saving rollout results (only used in eval_rollout mode).')
    
    # GNN Architecture arguments
    gnn_group = parser.add_argument_group('GNN Architecture Arguments')
    gnn_group.add_argument('--hidden_channels', type=int, default=32,
                        help='Number of hidden channels in GNN layers.')
    gnn_group.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate.')
    gnn_group.add_argument('--num_gnn_layers', type=int, default=2,
                        help='Number of GNN layers.')
    
    # GAT-specific arguments
    gat_group = parser.add_argument_group('GAT-specific Arguments (only used when gnn_type=gat)')
    gat_group.add_argument('--gat_heads', type=int, default=8,
                        help='Number of attention heads for GAT.')
    gat_group.add_argument('--out_heads', type=int, default=1,
                        help='Number of output heads for GAT.')
    
    # TransGNN-specific arguments
    trans_group = parser.add_argument_group('TransGNN-specific Arguments (only used when gnn_type=trans_gnn)')
    trans_group.add_argument('--use_bn', action='store_true',
                        help='Use layer normalization.')
    trans_group.add_argument('--dropedge', type=float, default=0.0,
                        help='Edge dropout rate for regularization.')
    trans_group.add_argument('--dropnode', type=float, default=0.0,
                        help='Node dropout rate for regularization.')
    trans_group.add_argument('--trans_heads', type=int, default=4,
                        help='Number of transformer heads.')
    trans_group.add_argument('--nb_random_features', type=int, default=30,
                        help='Number of random features.')
    trans_group.add_argument('--use_gumbel', action='store_true',
                        help='Use Gumbel softmax for message passing.')
    trans_group.add_argument('--use_residual', action='store_true',
                        help='Use residual connections for each GNN layer.')
    trans_group.add_argument('--nb_sample_gumbel', type=int, default=10,
                        help='Number of samples for Gumbel softmax sampling.')
    trans_group.add_argument('--temperature', type=float, default=0.25,
                        help='Temperature coefficient for softmax.')
    trans_group.add_argument('--reg_weight', type=float, default=0.1,
                        help='Weight for graph regularization.')
    trans_group.add_argument('--projection_matrix_type', type=bool, default=True,
                        help='Use projection matrix.')
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(f'{args.model_path}/{args.dataset}/{args.gnn_type}',
               exist_ok=True)
    if args.mode == 'eval_rollout':
        os.makedirs(f'{args.output_path}/{args.dataset}/{args.gnn_type}',
                   exist_ok=True)
    
    print_args(args)
    return args

if __name__ == '__main__':
    args = parse_arguments()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval_one_step(args)
    elif args.mode == 'eval_rollout':
        eval_rollout(args)
    else:
        raise ValueError("Unrecognized mode!")
