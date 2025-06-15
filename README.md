# Staircase Foveate Vision Project

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)


## Project Overview

Implementing human-like visual attention with high-resolution central vision and low-resolution peripheral processing.

### Installation

```bash
# Clone the repository
git clone https://github.com/wppqywq/foveate-vision
cd foveate-vision

# Create conda environment
conda create -n foveal-vision python=3.9 -y
conda activate foveal-vision

# Install dependencies
pip install torch torchvision opencv-python matplotlib tqdm tensorboard pillow
```

### Quick Demo

```bash
# Run setup and demonstration
python setup_phase1.py

# Train baseline model (3 glimpses, 30 epochs)
python experiments/train_baseline.py --epochs 30 --n_glimpses 3

# Compare with full resolution baseline
python experiments/train_baseline.py --epochs 30 --n_glimpses 3 --compare_full
```
### Resume Training Commands
```bash

# Resume from specific checkpoint:
python experiments/train_baseline.py --epochs 30 --resume logs/baseline_**/checkpoint_epoch_**.pth
# Auto-resume or extend from latest:
python experiments/train_baseline.py --epochs 30 --auto_resume
```

## Project Structure

```
foveate-vision/
├── fovea_lib/                # Core library
│   ├── transforms.py         # Foveal sampling implementation
│   ├── dataset_builder.py    # Multi-glimpse dataset builders
│   ├── baseline_model.py     # Fixed attention models
│   ├── metrics.py            # Efficiency evaluation metrics
│   └── __init__.py           # Package initialization
├── experiments/              # Experimental scripts
│   ├── train_baseline.py     # Training pipeline
│   ├── setup_phase1.py       # Setup and demo
│   └── debug_opencv.py       # Debugging utilities
├── logs/                     # Training logs and checkpoints
├── data/                     # Dataset storage
├── results/                  # Experimental results
└── tests/                    # tests
```

## Methodology

### Foveal Vision Architecture

```python
# Multi-resolution sampling strategy
Input Image (32×32×3)
    ↓ FovealTransform
Multiple Glimpses (N×16×16×3)    # High-res foveal regions  
    ↓ SharedResNetEncoder
Feature Vectors (N×512)
    ↓ Attention/Concatenation
Combined Features
    ↓ Classification
Predictions (10 classes)
```

### Key Design Principles

1. **Shared-weight processing**: All glimpses processed by same encoder (translation invariance)
2. **Fixed attention baseline**: Establishes performance benchmarks before dynamic attention
3. **Efficiency-first evaluation**: Novel metrics balancing accuracy and computational cost

### Efficiency Metrics

- **Information Efficiency (IE)**: `accuracy / n_glimpses` - measures information gain per fixation
- **Bandwidth Efficiency (BW)**: `pixels_processed / n_glimpses` - computational efficiency metric  
- **Temporal Efficiency (TE)**: `accuracy / processing_time` - real-time performance indicator


## References

todo