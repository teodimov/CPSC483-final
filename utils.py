# Standard library imports
import collections
import json
import os
import random
from typing import Dict, Any, Union, NamedTuple

# Third-party imports
import numpy as np
import torch
from texttable import Texttable
import pickle


# Type definitions
Stats = collections.namedtuple('Stats', ['mean', 'std'])

# Constants
INPUT_SEQUENCE_LENGTH = 6  # For calculating last 5 velocities
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_args(args) -> None:
    """Print arguments in a formatted table."""
    args_dict = vars(args)
    keys = sorted(args_dict.keys())
    
    table = Texttable()
    rows = [["Parameter", "Value"]]
    rows.extend(
        [k.replace("_", " ").capitalize(), args_dict[k]]
        for k in keys
    )
    
    table.add_rows(rows)
    print(table.draw())

def fix_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_kinematic_mask(particle_types: torch.Tensor) -> torch.Tensor:
    """Get boolean mask for kinematic particles."""
    return torch.eq(particle_types, KINEMATIC_PARTICLE_ID)

def _combine_std(std_x: float, std_y: float) -> float:
    """Combine standard deviations using root sum of squares."""
    return np.sqrt(std_x**2 + std_y**2)

def _read_metadata(data_path: str, split: str = 'train') -> Dict[str, Any]:
    """Read metadata from JSON and include positions."""
    metadata_path = os.path.join(data_path, 'metadata.json')
    positions_path = os.path.join(data_path, split, 'positions.pkl')  # Use the split directory

    print(f"Looking for metadata at: {metadata_path}")
    print(f"Looking for positions at: {positions_path}")

    with open(metadata_path, 'rt') as fp:
        metadata = json.loads(fp.read())
    
    # Load positions from the correct split directory
    if os.path.exists(positions_path):
        with open(positions_path, 'rb') as fp:
            metadata['positions'] = pickle.load(fp)
    else:
        raise FileNotFoundError(f"Positions file not found at: {positions_path}")
    
    return metadata


def compute_adaptive_radius(positions, gradients, base_radius):
    """
    Computes adaptive connectivity radius for each node.

    Args:
        positions (np.ndarray): Node positions [num_nodes, num_dims].
        gradients (np.ndarray): Gradient values for each node [num_nodes].
        base_radius (float): Base radius for connectivity.

    Returns:
        np.ndarray: Adaptive radius for each node [num_nodes].
    """
    grad_norm = gradients / max(gradients.max(), 1e-8)  # Normalize gradients safely
    adaptive_radius = base_radius * (1 + grad_norm)  # Scale base radius
    return adaptive_radius
