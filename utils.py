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

def _read_metadata(data_path: str) -> Dict[str, Any]:
    """Read metadata from JSON file."""
    metadata_path = os.path.join(data_path, 'metadata.json')
    with open(metadata_path, 'rt') as fp:
        return json.loads(fp.read())
