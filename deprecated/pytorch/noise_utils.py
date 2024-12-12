"""Methods to calculate input noise."""

import torch
from torch import Tensor
import math
from learned_simulator import time_diff
from typing import Union


def get_random_walk_noise_for_position_sequence(
    position_sequence: Tensor,
    noise_std_last_step: Union[float, Tensor]
) -> Tensor:
    """Returns random-walk noise in the velocity applied to the position.

    We want the noise scale in the velocity at the last step to be fixed.
    Because we are going to compose noise at each step using a random_walk:
    std_last_step**2 = num_velocities * std_each_step**2
    so to keep `std_last_step` fixed, we apply at each step:
    std_each_step = std_last_step / np.sqrt(num_input_velocities)

    Args:
        position_sequence: Tensor of shape [num_particles, sequence_length, num_dimensions]
            representing the position of particles across a sequence of time steps.
        noise_std_last_step: A scalar float or Tensor specifying the noise standard 
            deviation at the last step in terms of velocity.

    Returns:
        position_sequence_noise: Tensor of same shape as position_sequence
            representing the integrated random-walk noise added to the positions.
    """
    # Compute velocity sequence by finite differences
    velocity_sequence = time_diff(position_sequence)
    
    # Get number of velocity steps from the shape
    num_velocities = velocity_sequence.size(1)
    
    # Compute per-step noise standard deviation
    std_each_step = noise_std_last_step / math.sqrt(num_velocities)
    
    # Sample random velocity noise with proper scaling
    velocity_sequence_noise = torch.randn_like(
        velocity_sequence,
        dtype=position_sequence.dtype) * std_each_step
    
    # Apply random walk via cumulative sum
    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)
    
    # Integrate the noise in the velocity to positions
    # No noise added to first position since it's only used for first position change
    first_position_noise = torch.zeros_like(velocity_sequence_noise[:, 0:1])
    position_sequence_noise = torch.cat([
        first_position_noise,
        torch.cumsum(velocity_sequence_noise, dim=1)
    ], dim=1)
    
    return position_sequence_noise