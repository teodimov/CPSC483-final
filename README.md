# Learning to Simulate Physics

This project provides an implementation for learning to simulate particle-based physics, adapted from [Learn-to-Simulate](https://github.com/Emiyalzn/Learn-to-Simulate/tree/main).

## Overview

The system learns to simulate particle-based physics by using Graph Neural Networks (GNNs) to model particle interactions. It supports different types of simulations including water drops and water ramps.

## Features

- Multiple GNN architectures (e.g., GCN and GAT)
- Single-step and rollout predictions
- Support for different particle types
- Noise injection for robustness
- Checkpoint saving and loading
- Validation during training

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Elder453/CPSC483-final.git
cd CPSC483-final
```

2. Install dependencies:
```bash
conda env create -f environment.yml
```

3. Create necessary directories:
```bash
mkdir -p ./datasets
mkdir -p ./models
mkdir -p ./rollouts
```

## Dataset

1. Download a dataset (e.g., Water or WaterDropSample):
```bash
bash ./download_dataset.sh WaterDropSample ./datasets
```

## Usage

### Training

Train a model with default parameters:
```bash
python main.py --mode train --dataset WaterDropSample
```

For faster training with optimized parameters:
```bash
python main.py --mode train \
  --dataset WaterDropSample \
  --batch_size 2 \
  --max_episodes 100 \
  --test_step 500 \
  --lr 1e-4 \
  --noise_std 0.0003 \
  --message_passing_steps 5
```

### Evaluation

Evaluate single-step predictions:
```bash
python main.py --mode eval \
  --dataset WaterDropSample \
  --message_passing_steps 5 \
  --eval_split test
```

Generate trajectory rollouts:
```bash
python main.py --mode eval_rollout \
  --dataset WaterDropSample \
  --message_passing_steps 5 \
  --eval_split test
```

## Key Parameters

- `--mode`: Training or evaluation mode (`train`, `eval`, `eval_rollout`)
- `--dataset`: Name of dataset to use
- `--batch_size`: Number of samples per batch
- `--max_episodes`: Number of training episodes
- `--message_passing_steps`: Number of GNN message passing steps
- `--gnn_type`: Type of GNN to use (`gcn`, `gat`)
- `--noise_std`: Standard deviation of noise injection
- `--lr`: Learning rate

## Directory Structure

```
/CPSC483-final/
├── datasets/           # Dataset storage
│   └── WaterDropSample/
├── models/            # Saved model checkpoints
│   └── WaterDropSample/
└── rollouts/          # Generated rollouts
    └── WaterDropSample/
```

## Notes

- The code assumes CUDA availability. For CPU-only usage, modify device settings accordingly.
- Training time varies based on dataset size and computational resources.
- Best checkpoints are automatically saved during training.
- Parameters may need adjustment based on your specific use case.

## Acknowledgments

This PyTorch implementation is adapted from [Learn-to-Simulate](https://github.com/Emiyalzn/Learn-to-Simulate/tree/main). We thank the original authors for their work.

The original work was done by [DeepMind](https://github.com/deepmind/deepmind-research), written in TensorFlow and published at ICML2020.

```shell
@article
{DBLP:journals/corr/abs-2002-09405,
  author    = {Alvaro Sanchez{-}Gonzalez and
               Jonathan Godwin and
               Tobias Pfaff and
               Rex Ying and
               Jure Leskovec and
               Peter W. Battaglia},
  title     = {Learning to Simulate Complex Physics with Graph Networks},
  journal   = {CoRR},
  volume    = {abs/2002.09405},
  year      = {2020},
  url       = {https://arxiv.org/abs/2002.09405},
  eprinttype = {arXiv},
  eprint    = {2002.09405},
  timestamp = {Mon, 02 Mar 2020 16:46:06 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2002-09405.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
