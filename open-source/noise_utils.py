import torch
from typing import Union, Tuple
import numpy as np
from learned_simulator import time_diff

def get_random_walk_noise_for_position_sequence(
    position_sequence: torch.Tensor,
    noise_std_last_step: float
) -> torch.Tensor:
    """Generate random walk noise to apply to a position sequence.

    This function generates noise that follows a random walk pattern in velocity
    space, which is then integrated to get position noise. The noise standard
    deviation at the last step is controlled by noise_std_last_step.

    Args:
        position_sequence: Tensor of particle positions over time
            Shape: [num_particles, num_timesteps, num_dimensions]
        noise_std_last_step: Target standard deviation of noise at the last step

    Returns:
        Tensor of position noise with same shape as position_sequence
            Shape: [num_particles, num_timesteps, num_dimensions]

    Note:
        The noise is generated as a random walk in velocity space and then
        integrated to get position noise. The noise at each step is scaled
        so that the final step has the desired standard deviation.
    """
    # Calculate velocity sequence from positions
    velocity_sequence = time_diff(position_sequence)

    # Calculate number of velocity steps
    num_velocities = velocity_sequence.shape[1]

    # Generate velocity noise
    # Scale is set so that accumulated noise at last step has desired std
    step_noise_std = noise_std_last_step / np.sqrt(num_velocities)
    velocity_sequence_noise = torch.normal(
        mean=0.0,
        std=step_noise_std,
        size=velocity_sequence.shape,
        dtype=position_sequence.dtype,
        device=position_sequence.device  # Ensure noise is on same device
    )

    # Accumulate velocity noise over time
    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

    # Integrate velocity noise to get position noise
    # Start with zero noise at first position
    initial_position_noise = torch.zeros_like(velocity_sequence_noise[:, 0:1])
    position_sequence_noise = torch.cat([
        initial_position_noise,
        torch.cumsum(velocity_sequence_noise, dim=1)
    ], dim=1)

    return position_sequence_noise

def validate_noise_inputs(
    position_sequence: torch.Tensor,
    noise_std_last_step: float
) -> None:
    """Validate inputs to noise generation function.

    Args:
        position_sequence: Position sequence tensor
        noise_std_last_step: Noise standard deviation

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(position_sequence, torch.Tensor):
        raise ValueError("position_sequence must be a torch.Tensor")
    
    if position_sequence.dim() != 3:
        raise ValueError(
            f"position_sequence must have 3 dimensions, got {position_sequence.dim()}"
        )
    
    if noise_std_last_step < 0:
        raise ValueError(
            f"noise_std_last_step must be non-negative, got {noise_std_last_step}"
        )
