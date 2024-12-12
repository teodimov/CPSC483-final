import os
import pickle
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from learned_simulator import LearnedSimulator
from noise_utils import get_random_walk_noise_for_position_sequence
from dataloader import OneStepDataset, RolloutDataset, one_step_collate
from rollout import rollout
from utils import fix_seed, _combine_std, _read_metadata
from utils import *

def _get_simulator(
    model_kwargs: dict,
    metadata: dict,
    acc_noise_std: float,
    vel_noise_std: float,
    args
) -> 'LearnedSimulator':
    """Initialize simulator with proper normalization statistics.
    
    Args:
        model_kwargs: Dictionary containing model parameters like latent_size, mlp_hidden_size etc.
        metadata: Dictionary containing simulation metadata including dimensions, bounds etc.
        acc_noise_std: Standard deviation of acceleration noise
        vel_noise_std: Standard deviation of velocity noise
        args: Additional arguments passed from command line
        
    Returns:
        LearnedSimulator: Initialized simulator instance with proper normalization
    """
    cast = lambda v: np.array(v, dtype=np.float32)
    
    acceleration_stats = Stats(
        cast(metadata['acc_mean']),
        _combine_std(cast(metadata['acc_std']), acc_noise_std))
    
    velocity_stats = Stats(
        cast(metadata['vel_mean']),
        _combine_std(cast(metadata['vel_std']), vel_noise_std))
    
    normalization_stats = {'acceleration': acceleration_stats,
                          'velocity': velocity_stats}
    
    if 'context_mean' in metadata:
        context_stats = Stats(
            cast(metadata['context_mean']), cast(metadata['context_std']))
        normalization_stats['context'] = context_stats

    simulator = LearnedSimulator(
        num_dimensions=metadata['dim'],
        connectivity_radius=metadata['default_connectivity_radius'],
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
        if file.startswith('step'):
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
        if file.startswith('step'):
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
            - max_episodes: Number of training episodes
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
    valid_dataset = OneStepDataset(args.dataset, 'valid')
    num_valid_samples = len(valid_dataset)
    test_valid_samples = min(2000, num_valid_samples)  # number of valid samples to be used

    # Model initialization
    metadata = _read_metadata(data_path=f"datasets/{args.dataset}")
    model_kwargs = dict(
        latent_size=128,
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=args.message_passing_steps,
    )

    simulator = _get_simulator(
        model_kwargs=model_kwargs,
        metadata=metadata,
        vel_noise_std=args.noise_std,
        acc_noise_std=args.noise_std,
        args=args
    )
    
    # Optimization setup
    min_lr = 1e-6
    decay_lr = args.lr - min_lr
    optimizer = torch.optim.Adam(
        simulator.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    mse_loss = F.mse_loss
    best_one_step_loss = float("inf")
    time_step = 0

    # Training loop
    print("################### Begin Training #######################")
    for episode in range(args.max_episodes + 1):
        print(f'Episode: {episode}/{args.max_episodes}\n')
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
            simulator.train()
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

            # Calculate masked loss
            loss = (pred_acceleration[non_kinematic_mask] - 
                   target_acceleration[non_kinematic_mask]) ** 2
            num_non_kinematic = torch.sum(non_kinematic_mask.to(torch.float32))
            loss = torch.sum(loss) / torch.sum(num_non_kinematic)

            # Optimization step
            loss.mean().backward()
            optimizer.step()
            
            # Learning rate decay
            decay_lr = decay_lr * (0.1 ** (1/5e6))
            for param_group in optimizer.param_groups:
                param_group['lr'] = decay_lr + min_lr

            # Evaluation step
            simulator.eval()
            with torch.no_grad():
                predicted_next_position = simulator(
                    position_sequence=features['positions'],
                    n_particles_per_example=features['n_particles_per_example'],
                    particle_types=features['particle_types'],
                    global_context=features.get('step_context')
                )

                loss_mse = mse_loss(pred_acceleration, target_acceleration)
                one_step_position_mse = mse_loss(predicted_next_position, target_next_position)
                if time_step % 100 == 0:  # Print every 100 steps
                    print(f"[Step {time_step}] "
                          f"Loss MSE: {loss_mse:.4f}, "
                          f"Position MSE: {one_step_position_mse*1e9:.4f}e-9")
                time_step += 1

                # Periodic validation
                if time_step % args.test_step == 0:
                    print("################### Begin Evaluate One Step #######################")
                    total_loss = []
                    
                    # Create validation dataloader with subset of samples
                    test_indices = torch.tensor(np.random.choice(num_valid_samples, test_valid_samples, False))
                    valid_dataloader = DataLoader(
                        Subset(valid_dataset, test_indices),
                        collate_fn=one_step_collate,
                        batch_size=args.batch_size,
                        shuffle=True
                    )

                    # Validation loop
                    for val_features, val_labels in valid_dataloader:
                        val_labels = val_labels.to(device)
                        for key in ['positions', 'particle_types', 'n_particles_per_example']:
                            val_features[key] = val_features[key].to(device)
                        if 'step_context' in val_features:
                            val_features['step_context'] = val_features['step_context'].to(device)
                        val_target_next_position = val_labels

                        # Compute validation predictions
                        val_predicted_next_position = simulator(
                            position_sequence=val_features['positions'],
                            n_particles_per_example=val_features['n_particles_per_example'],
                            particle_types=val_features['particle_types'],
                            global_context=val_features.get('step_context')
                        )
                        
                        # Calculate validation loss
                        val_one_step_position_mse = mse_loss(val_predicted_next_position, val_target_next_position)
                        total_loss.append(val_one_step_position_mse)

                    # Save best model checkpoint
                    average_one_step_loss = torch.tensor(total_loss).mean().item()
                    if average_one_step_loss < best_one_step_loss:
                        best_one_step_loss = average_one_step_loss
                        checkpoint_path = (f'{args.model_path}/{args.dataset}/{args.gnn_type}/'
                                         f'step_{time_step}_best_one_step_loss_{average_one_step_loss * 1e9:.2f}e-9.pt')
                        print(f"Saving the best checkpoint, episode: {episode}, "
                              f"one step position loss: {average_one_step_loss * 1e9:.2f}e-9.")
                        torch.save(simulator.state_dict(), checkpoint_path)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Learning to Simulate.")
    
    # Simulation settings
    parser.add_argument('--mode', default='train',
                        choices=['train', 'eval', 'eval_rollout'],
                        help='Train model, one step evaluation or rollout evaluation.')
    parser.add_argument('--eval_split', default='test',
                        choices=['train', 'valid', 'test'],
                        help='Split to use when running evaluation.')
    parser.add_argument('--dataset', default="Water", type=str,
                        help='The dataset directory.')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='The batch size.')
    parser.add_argument('--max_episodes', default=10000, type=int,
                        help='Number of steps of training.')
    parser.add_argument('--test_step', default=5000, type=int,
                        help='Number of saving step.')
    parser.add_argument('--noise_std', default=0.0003, type=float,
                        help='The std deviation of the noise.')
    parser.add_argument('--model_path', default="models", type=str,
                        help='The path for saving checkpoints of the model.')
    parser.add_argument('--output_path', default="rollouts", type=str,
                        help='The path for saving outputs (e.g. rollouts).')
    parser.add_argument('--gnn_type', default='gcn', choices=['gcn', 'gat', 'trans_gnn', 'interaction_net'],
                        help='The GNN to be used as processor.')
    parser.add_argument('--message_passing_steps', default=10, type=int, help='number of GNN message passing steps.')

    # GNN settings
    parser.add_argument('--seed', type=int, default=483)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_gnn_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')
    parser.add_argument('--projection_matrix_type', type=bool, default=True,
                        help='use projection matrix or not')

    # TransGNN settings
    parser.add_argument('--use_bn', action='store_true',
                        help='use layernorm')
    parser.add_argument('--dropedge', type=float, default=0.0,
                        help='dropedge for regularization')
    parser.add_argument('--dropnode', type=float, default=0.0,
                        help='dropedge for regularization')
    parser.add_argument('--trans_heads', type=int, default=4)
    parser.add_argument('--nb_random_features', type=int,
                        default=30, help='number of random features')
    parser.add_argument('--use_gumbel', action='store_true',
                        help='use gumbel softmax for message passing')
    parser.add_argument('--use_residual', action='store_true',
                        help='use residual link for each GNN layer')
    parser.add_argument('--nb_sample_gumbel', type=int, default=10,
                        help='num of samples for gumbel softmax sampling')
    parser.add_argument('--temperature', type=float, default=0.25,
                        help='temp coefficient for softmax')
    parser.add_argument('--reg_weight', type=float, default=0.1,
                        help='weight for graph reg')
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(f'{args.model_path}/{args.dataset}/{args.gnn_type}',
               exist_ok=True)
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
