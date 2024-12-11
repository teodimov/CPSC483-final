# reading_utils.py
"""Utilities for reading converted Learning Complex Physics data from .npz files."""

import os
import numpy as np

def load_data_from_npz(data_path: str, split: str):
  """Loads the trajectories from a pre-converted .npz file.

  Args:
    data_path: Path to the dataset directory containing {split}_data.npz files.
    split: One of 'train', 'valid', 'test'.

  Returns:
    A NumPy array of trajectories. Each element is a dictionary with keys:
      'particle_type': np.ndarray [num_particles]
      'position': np.ndarray [sequence_length+1, num_particles, dim]
      and optionally 'step_context': np.ndarray [sequence_length+1, context_feat_dim]
  """
  file_path = os.path.join(data_path, f"{split}_data.npz")
  loaded = np.load(file_path, allow_pickle=True)
  trajectories = loaded['trajectories']
  return trajectories


def split_trajectory(trajectory: dict, window_length: int = 7):
  """Splits a single trajectory into overlapping windows.

  Args:
    trajectory: A dictionary with at least:
      'position': np.ndarray [sequence_length+1, num_particles, dim]
      'particle_type': np.ndarray [num_particles]
      optionally 'step_context': np.ndarray [sequence_length+1, context_feat_dim]
    window_length: The length of each trajectory window.

  Returns:
    A list of dictionaries, each representing a window:
      {
        'position': [window_length, num_particles, dim]
        'particle_type': [num_particles]
        'step_context': [window_length, context_feat_dim] (if available)
      }

    The number of returned windows is (sequence_length+1 - window_length + 1).
  """
  pos = trajectory['position']
  trajectory_length = pos.shape[0]
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
