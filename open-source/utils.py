import numpy as np
from texttable import Texttable
import random
import torch
import collections
import os
import json
from typing import Dict, Any, Union, NamedTuple

# Type definitions
Stats = collections.namedtuple('Stats', ['mean', 'std'])

# Constants
INPUT_SEQUENCE_LENGTH = 6  # For calculating last 5 velocities
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else 
                     #"mps" if torch.backends.mps.is_available() else 
                     "cpu")

def print_args(args):
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

def fix_seed(seed):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_kinematic_mask(particle_types):
    """Get boolean mask for kinematic particles."""
    return torch.eq(particle_types, KINEMATIC_PARTICLE_ID)

def _combine_std(std_x, std_y):
    """Combine standard deviations using root sum of squares."""
    return np.sqrt(std_x**2 + std_y**2)

def _read_metadata(data_path):
    """Read metadata from JSON file."""
    metadata_path = os.path.join(data_path, 'metadata.json')
    with open(metadata_path, 'rt') as fp:
        return json.loads(fp.read())
