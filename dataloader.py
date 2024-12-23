import torch
import pickle
import os
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

INPUT_SEQUENCE_LENGTH = 6

@dataclass
class Features:
    positions: torch.Tensor
    particle_types: torch.Tensor
    step_context: Optional[torch.Tensor] = None

class NCDataset:
    """Single-graph dataset container."""
    def __init__(self, name: str):
        self.name = name
        self.graph = {}
        self.label = None

    def __getitem__(self, idx: int) -> Tuple[dict, Optional[object]]:
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self) -> int:
        return 1

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'

class BaseDataset(Dataset):
    """Base class for particle simulation datasets."""
    def __init__(self, dataset: str = 'Water', split: str = 'train'):
        positions, particle_types, step_context = self._load_data(dataset, split)
        self.features = self._prepare_features(positions, particle_types, step_context)
        self.gradients = self.features['gradients']  # Store gradients separately


    def _load_data(self, dataset: str, split: str) -> Tuple[List, List, Optional[List]]:
        """Loads data from pickle files."""
        datapath = f"datasets/{dataset}/{split}"
        with open(os.path.join(datapath, "positions.pkl"), 'rb') as f:
            positions = pickle.load(f)
        with open(os.path.join(datapath, "particle_types.pkl"), 'rb') as f:
            particle_types = pickle.load(f)
        
        context_file = os.path.join(datapath, "step_context.pkl")
        step_context = None
        if os.path.exists(context_file):
            with open(context_file, 'rb') as f:
                step_context = pickle.load(f)
        
        return positions, particle_types, step_context

    def _prepare_features(
        self,
        positions: List,
        particle_types: List,
        step_context: Optional[List]
    ) -> Dict[str, List[torch.Tensor]]:
        """Converts numpy arrays to torch tensors and computes gradients."""
        features = {
            'positions': list(map(torch.tensor, positions)),
            'particle_types': list(map(torch.tensor, particle_types))
        }
        if step_context is not None:
            features['step_context'] = list(map(torch.tensor, step_context))
        
        # Compute gradients for each position set
        gradients = []
        for pos in positions:
            pos_tensor = torch.tensor(pos)  # Shape: (T, N, dim)
            # Compute gradients (difference between positions along spatial axes)
            diff = torch.diff(pos_tensor, dim=0)  # Shape: (T-1, N, dim)
            grad_magnitude = torch.norm(diff, dim=-1).mean(dim=0)  # Mean over time, Shape: (N,)
            gradients.append(grad_magnitude)
        
        features['gradients'] = gradients  # Add gradients to features
        return features




    def __len__(self) -> int:
        return len(self.features['positions'])

class OneStepDataset(BaseDataset):
    """Dataset for single-step predictions."""
    def __init__(self, dataset: str = 'Water', split: str = 'train'):
        super().__init__(dataset, split)
        self.features = split_trajectory(self.features, INPUT_SEQUENCE_LENGTH + 1)

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        feature = {
            'positions': self.features['positions'][index],
            'particle_types': self.features['particle_types'][index]
        }
        if 'step_context' in self.features:
            feature['step_context'] = self.features['step_context'][index]
        return prepare_inputs(feature)

class RolloutDataset(BaseDataset):
    """Dataset for trajectory rollouts."""
    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        feature = {
            'positions': self.features['positions'][index],
            'particle_types': self.features['particle_types'][index]
        }
        if 'step_context' in self.features:
            feature['step_context'] = self.features['step_context'][index]
        return prepare_rollout_inputs(feature)

def split_trajectory(features: Dict[str, List[torch.Tensor]], 
                    window_length: int = 7) -> Dict[str, List[torch.Tensor]]:
    """Splits trajectory into sliding windows."""
    trajectory_length = features['positions'][0].shape[0]
    input_trajectory_length = trajectory_length - window_length + 1
    
    model_input_features = {
        'particle_types': [],
        'positions': []
    }
    if 'step_context' in features:
        model_input_features['step_context'] = []

    for i in range(len(features['positions'])):
        for idx in range(input_trajectory_length):
            model_input_features['positions'].append(
                features['positions'][i][idx:idx + window_length])
            model_input_features['particle_types'].append(
                features['particle_types'][i])
            if 'step_context' in features:
                model_input_features['step_context'].append(
                    features['step_context'][i][idx:idx + window_length])

    return model_input_features

def one_step_collate(batch: List[Tuple]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Collates batches for training."""
    output_dict = {
        'positions': torch.cat([sample['positions'] for sample, _ in batch], dim=0),
        'n_particles_per_example': torch.cat([sample['n_particles_per_example'] 
                                            for sample, _ in batch], dim=0),
        'particle_types': torch.cat([sample['particle_types'] for sample, _ in batch], dim=0)
    }
    
    if 'step_context' in batch[0][0]:
        output_dict['step_context'] = torch.cat([sample['step_context'] 
                                               for sample, _ in batch], dim=0)
    
    output_target = torch.cat([target for _, target in batch], dim=0)
    return output_dict, output_target

def prepare_inputs(tensor_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], 
                                                                torch.Tensor]:
    """Prepares input tensors for the model."""
    pos = torch.transpose(tensor_dict['positions'], 0, 1)
    target_position = pos[:, -1]
    
    output_dict = {
        'positions': pos[:, :-1],
        'n_particles_per_example': torch.tensor(pos.shape[0]).unsqueeze(0),
        'particle_types': tensor_dict['particle_types']
    }
    
    if 'step_context' in tensor_dict:
        output_dict['step_context'] = tensor_dict['step_context'][-2].unsqueeze(0)
    
    return output_dict, target_position

def prepare_rollout_inputs(features: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], 
                                                                     torch.Tensor]:
    """Prepares an input trajectory for rollout."""
    pos = torch.transpose(features['positions'], 0, 1)
    
    output_dict = {
        'positions': pos[:, :-1],
        'particle_types': features['particle_types'],
        'n_particles_per_example': torch.tensor(pos.shape[0]).unsqueeze(0),
        'is_trajectory': torch.tensor([True], dtype=torch.bool)
    }
    
    if 'step_context' in features:
        output_dict['step_context'] = features['step_context']
    
    return output_dict, pos[:, -1]
