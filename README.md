# Foveal Attention Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

TODO: 
- Haven't successes in debugging RL algorithm. Only structures to demo.
- Dynamic sampling grid to be implemented.
- Final classification task.

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


## Reference

- Cheung et al. (2016): "Emergence of foveal image sampling from learning to attend in visual scenes" (**Theory**)
- Mnih et al. (2014): "Recurrent Models of Visual Attention" (**Architecture**)

    Official implementation: [GitHub](https://github.com/schefferac2020/FovialEmergence)

