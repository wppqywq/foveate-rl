# Foveal Attention Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

Draft implementation of **"Emergence of foveal image sampling from learning to attend in visual scenes"** (Cheung et al. 2016). Demonstrates how foveal attention patterns emerge naturally from reinforcement learning without explicit foveal bias.

## Quick Start

```bash
# Install dependencies
pip install torch torchvision matplotlib scipy tqdm

# Clone repository
git clone https://github.com/wppqywq/foveal-vision
cd foveal-attention

# Run demo
python demo_rl_attention.py

# Train full model
python train_rl_attention.py --epochs 50 --dataset cifar10
```

## Project Structure

```
foveal-attention/
├── fovea_lib/              # Core library
│   ├── transforms.py       # Foveal sampling implementation
│   ├── rl_attention.py     # RL attention model + REINFORCE
│   ├── dataset_builder.py  # Data loading utilities  
│   ├── metrics.py          # Emergence analysis metrics
│   └── __init__.py         
├── train_rl_attention.py   # Main training script
├── demo_rl_attention.py    # Quick demonstration
└── results/                # Training outputs
```

## Architecture

The model learns attention policies through REINFORCE:

1. **Glimpse Network**: Encodes foveated image patches
2. **Location Network**: Learns where to attend next
3. **Recurrent Core**: Maintains attention state across glimpses
4. **Classifier**: Final prediction based on attended regions


## Usage Examples

### Basic Training
```bash
python train_rl_attention.py --dataset cifar10 --epochs 50
```

### Custom Configuration
```bash
python train_rl_attention.py \
    --dataset mnist \
    --epochs 100 \
    --batch_size 64 \
    --max_glimpses 8 \
    --lr 0.001
```

### Analysis
```python
from fovea_lib import EmergenceAnalyzer, TrainingMonitor

# Analyze attention patterns
analyzer = EmergenceAnalyzer()
foveal_score = analyzer.compute_foveal_score(attention_patterns)
print(f"Foveal score: {foveal_score:.3f}")
```

## Key Papers

- Cheung et al. (2016): "Emergence of foveal image sampling from learning to attend in visual scenes"
- Mnih et al. (2014): "Recurrent Models of Visual Attention"

