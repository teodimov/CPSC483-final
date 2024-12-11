# noise_utils.py

import torch
from torch import Tensor
import math
from learned_simulator import time_diff

def get_random_walk_noise_for_position_sequence(
    position_sequence: Tensor, noise_std_last_step: float
) -> Tensor:
    """
    Returns random-walk noise in the velocity applied to the position.

    Args:
      position_sequence: A tensor of shape [num_particles, sequence_length, num_dimensions]
        representing the position of particles across a sequence of time steps.
      noise_std_last_step: A scalar float specifying the noise standard deviation
        at the last step in terms of velocity.

    Returns:
      position_sequence_noise: A tensor of the same shape as position_sequence that
        represents the integrated random-walk noise added to the positions.
    """

    # Compute the velocity sequence by finite differences:
    # velocity_sequence is [num_particles, sequence_length-1, num_dimensions]
    velocity_sequence = time_diff(position_sequence)

    # Compute the number of velocity steps
    # num_velocities = sequence_length - 1 (the second dimension of velocity_sequence)
    num_velocities = velocity_sequence.size(1)

    # Compute noise standard deviation per step so that at the last step the noise
    # matches noise_std_last_step when integrated over all steps.
    # std_each_step = noise_std_last_step / sqrt(num_velocities)
    std_each_step = noise_std_last_step / math.sqrt(num_velocities)

    # Sample random velocity noise
    # Shape is the same as velocity_sequence
    # torch.randn_like gives a tensor with the same shape and type.
    velocity_sequence_noise = torch.randn_like(velocity_sequence) * std_each_step

    # Apply the random walk by cumulative summation along the time dimension (dim=1)
    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

    # Integrate the velocity noise into position noise with Euler integration (dt=1).
    # The first position (at time step 0) gets no noise (since no previous velocity),
    # and subsequent positions accumulate noise.
    # Insert a zero vector for the first position:
    first_step_noise = torch.zeros_like(velocity_sequence_noise[:, 0:1, :])
    # Now accumulate over time:
    position_sequence_noise = torch.cat([first_step_noise, torch.cumsum(velocity_sequence_noise, dim=1)], dim=1)

    return position_sequence_noise
