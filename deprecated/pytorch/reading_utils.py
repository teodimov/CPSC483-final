"""Utilities for reading converted Learning Complex Physics data from NPZ files."""

import os
import numpy as np
from typing import Dict, List, Any

class DataLoadError(Exception):
    """Custom exception for data loading errors."""
    pass

def load_data_from_npz(data_path: str, split: str) -> List[Dict[str, np.ndarray]]:
    """Loads the trajectories from a pre-converted NPZ file.
    
    Args:
        data_path: Path to the dataset directory containing {split}_data.npz files
        split: One of 'train', 'valid', 'test'
    
    Returns:
        List of trajectory dictionaries, each containing:
            - particle_type: [num_particles]
            - position: [sequence_length+1, num_particles, dim]
            - step_context: [sequence_length+1, context_feat_dim] (optional)
            
    Raises:
        DataLoadError: If file not found or data format is invalid
    """
    file_path = os.path.join(data_path, f"{split}_data.npz")
    
    if not os.path.exists(file_path):
        raise DataLoadError(f"Data file not found: {file_path}")
        
    try:
        loaded = np.load(file_path, allow_pickle=True)
        trajectories = loaded['trajectories']
        
        # Validate data format
        for traj in trajectories:
            if not isinstance(traj, dict):
                raise DataLoadError("Invalid trajectory format")
            if 'position' not in traj or 'particle_type' not in traj:
                raise DataLoadError("Missing required fields in trajectory")
                
        return trajectories
        
    except Exception as e:
        raise DataLoadError(f"Error loading data: {str(e)}")

def split_trajectory(
    trajectory: Dict[str, np.ndarray],
    window_length: int = 7
) -> List[Dict[str, np.ndarray]]:
    """Splits a single trajectory into overlapping windows.
    
    Args:
        trajectory: Dictionary containing:
            - position: [sequence_length+1, num_particles, dim]
            - particle_type: [num_particles]
            - step_context: [sequence_length+1, context_feat_dim] (optional)
        window_length: Length of each trajectory window
    
    Returns:
        List of dictionaries, each containing:
            - position: [window_length, num_particles, dim]
            - particle_type: [num_particles]
            - step_context: [window_length, context_feat_dim] (if available)
            
    Raises:
        ValueError: If window_length is invalid or trajectory format is incorrect
    """
    if window_length < 1:
        raise ValueError("window_length must be positive")
        
    pos = trajectory['position']
    if len(pos.shape) != 3:
        raise ValueError("Position data must have shape [sequence_length+1, num_particles, dim]")
        
    trajectory_length = pos.shape[0]
    if trajectory_length < window_length:
        raise ValueError("Trajectory length must be >= window_length")
        
    input_trajectory_length = trajectory_length - window_length + 1
    windows = []
    has_context = 'step_context' in trajectory
    
    for idx in range(input_trajectory_length):
        window_data = {
            'position': pos[idx:idx + window_length],
            'particle_type': trajectory['particle_type']
        }
        if has_context:
            window_data['step_context'] = trajectory['step_context'][idx:idx + window_length]
        windows.append(window_data)
        
    return windows

def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """Validates that the metadata contains required fields.
    
    Args:
        metadata: Dictionary of metadata
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_fields = ['sequence_length', 'dim']
    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Missing required metadata field: {field}")
    return True