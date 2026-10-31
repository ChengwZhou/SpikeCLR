# SpikeCLR: Self-Supervised Contrastive Learning for Visual Representations with Spiking Neural Networks

**[ICML 2026]** Official implementation of the paper:

> **SpikeCLR: Self-Supervised Contrastive Learning for Visual Representations with Spiking Neural Networks**
> Accepted at the 43rd International Conference on Machine Learning (ICML 2026)
> [[Paper]](https://icml.cc/virtual/2026/poster/61778)

---

## Overview

This repository contains the official code for self-supervised representation learning on **Spiking Neural Networks (SNNs)** using a temporal-aware Barlow Twins objective. Key contributions include:

- **Temporal contrastive loss**: Cross-temporal and boundary-aligned temporal (BTL) loss variants that exploit the temporal dimension unique to SNNs.
- **MixedLIF activation**: A hybrid spiking/continuous activation function that enables contrastive learning with spike-based representations.
- **Neuromorphic data support**: Specialized training pipeline for event-camera datasets (DVS128 Gesture, N-Caltech101, ASL-DVS) with configurable timesteps.
- **Comprehensive evaluation**: Linear probing, semi-supervised fine-tuning (1%/10% labels), and transfer learning.

The repository is organized into two modules:

| Directory | Description |
|-----------|-------------|
| `static-SNN/` | Main framework: SSL training on CIFAR10, ImageNet, TinyImageNet with ResNet/ViT backbones |
| `dvs-SNN/` | Specialized pipeline for neuromorphic event-camera datasets |

---

## Requirements

### Environment

- Python >= 3.8
- CUDA >= 11.3 (for GPU training)
- PyTorch >= 1.12 (tested with PyTorch 1.13 and 2.0)

### Installation

```bash
git clone https://github.com/ChengwZhou/SpikeCLR.git
cd SpikeCLR
pip install -r requirements.txt
```

### Core dependencies (`requirements.txt`)

```
torch>=1.12
torchvision>=0.13
numpy
pyyaml
matplotlib
spikingjelly        # for neuromorphic dataset loading
tensorboard         # for training logging (dvs-SNN)
```

> **Note:** `spikingjelly` is only required for the `dvs-SNN` module.
> Install it via: `pip install spikingjelly`

---

## Repository Structure

```
SpikeCLR/
├── README.md
├── requirements.txt
│
├── static-SNN/                   # Main SSL framework
│   ├── main.py                   # Main training entry point (DDP)
│   ├── BartonTwins.py            # ANN Barlow Twins wrapper
│   ├── BartonTwins_spiking.py    # Spiking Barlow Twins wrapper
│   ├── loss.py                   # Loss functions (BarlowTwins, SimCLR, VICReg + temporal variants)
│   ├── model.py                  # Optimizer, scheduler, checkpointing utilities
│   ├── dataset.py                # Dataset utilities (TinyImageNet)
│   ├── linear_evaluation.py      # Linear probing on frozen backbone
│   ├── finetune_semi_supervised_learning.py  # Semi-supervised fine-tuning
│   ├── config/
│   │   └── config.yaml           # Default hyperparameters
│   ├── modules/                  # Network architectures
│   │   ├── spike_layer.py        # LIF, MixedLIF, LIFt spiking neurons
│   │   ├── resnet_spiking.py     # Spiking ResNet (18/34/50)
│   │   ├── vit.py                # Vision Transformer
│   │   └── transformations/      # Data augmentation
│   └── utils/                    # Config loading, weight utils
│
└── dvs-SNN/                      # Neuromorphic DVS pipeline
    ├── train.py                  # Distributed training (torchrun)
    ├── model.py                  # ResNet-SNN for DVS data
    ├── spiking_layer.py          # Spiking neuron implementations
    ├── barlowtwins.py            # BarlowTwins wrapper (DVS)
    ├── loss.py                   # BarlowTwinsLoss with temporal support
    ├── data_loaders.py           # DVS augmentation pipeline and dataset loaders
    ├── evaluation.py             # Linear evaluation
    ├── utils.py                  # Distributed utilities and metric logging
    └── autoaugment.py            # Auto-augmentation strategies
```

---

## Usage

### Standard Vision Datasets (CIFAR10 / ImageNet)

> All commands run from `static-SNN/`

**Warning:** Boolean flags must be passed as `--flag=True` or left empty `--flag=` (not `--flag False`) due to argparse parsing.

#### CIFAR10 — SNN with MixedLIF + 1st-last temporal loss (recommended)

```bash
# SSL pre-training
CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --gpus=2 --batch_size=256 \
  --model="resnet34" --dataset="CIFAR10" --n_classes=10 \
  --lr=1e-3 --spiking=True --temporal_loss=True \
  --cross_temporal=True --simplified_loss=True \
  --sync_norm=True --epochs=1000 --projection_dim=1024 \
  --act_func="MixedLIF" \
  --model_path="save/cifar10/resnet34_spk_1l_mixlif"

# Linear evaluation
python linear_evaluation.py \
  --model="resnet34" --dataset="CIFAR10" --n_classes=10 \
  --epoch_num=1000 --spiking=true --sync_norm=True \
  --act_func="MixedLIF" \
  --model_path="save/cifar10/resnet34_spk_1l_mixlif"

# Semi-supervised fine-tuning (1% labels)
python finetune_semi_supervised_learning.py \
  --eval_proportion=0.01 --logistic_epochs=60 \
  --model="resnet34" --dataset="CIFAR10" --n_classes=10 \
  --epoch_num=1000 --spiking=true --sync_norm=True \
  --model_path="save/cifar10/resnet34_spk_1l_mixlif"
```

#### ImageNet — ResNet50 SNN

```bash
# SSL pre-training
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
  --gpus=4 --batch_size=40 \
  --model="resnet50" --dataset="ImageNet" --n_classes=1000 \
  --lr=1e-3 --spiking=True --temporal_loss=True \
  --cross_temporal=True --simplified_loss=True \
  --sync_norm=True --epochs=1000 --projection_dim=8192 \
  --act_func="MixedLIF" \
  --model_path="save/imagenet/resnet50_spk_1l_mixlif" \
  --dataset_dir /path/to/imagenet

# Linear evaluation
python linear_evaluation.py \
  --model="resnet50" --dataset="ImageNet" --n_classes=1000 \
  --epoch_num=1000 --spiking=true --sync_norm=True \
  --act_func="MixedLIF" \
  --model_path="save/imagenet/resnet50_spk_1l_mixlif"
```

#### ANN Baseline (non-spiking)

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --gpus=2 --batch_size=256 \
  --model="resnet34" --dataset="CIFAR10" --n_classes=10 \
  --lr=1e-3 --epochs=1000 --projection_dim=1024 \
  --model_path="save/cifar10/resnet34"
```

---

### Neuromorphic Data (DVS128 Gesture)

> All commands run from `dvs-SNN/`
> Requires `spikingjelly` and the [DVS128 Gesture dataset](https://research.ibm.com/interactive/dvsgesture/).

#### Supported Datasets

| Dataset | Classes | Resolution | Split |
|---------|---------|------------|-------|
| DVS128 Gesture | 11 | 128×128 | Built-in train/test |
| N-Caltech101 | 101 | 240×180 | 90/10 random split |
| ASL-DVS | 24 | 180×240 → 128×128 | 90/10 random split |

#### SSL Pre-training (DVS128 Gesture, T=16, 4 GPUs)

```bash
# 1st-last temporal loss (BTL — simplified)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  --dataset /path/to/dvs128-gesture \
  --dataset-name dvs128-gesture \
  -b 80 --workers 4 --T 16 \
  --epochs 500

# Full cross-temporal loss (all T×T pairs)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  --dataset /path/to/dvs128-gesture \
  --dataset-name dvs128-gesture \
  -b 80 --workers 4 --T 16 \
  --epochs 500 --cross-temporal
```

#### Linear Evaluation

```bash
python evaluation.py \
  --dataset /path/to/dvs128-gesture \
  --dataset-name dvs128-gesture \
  --time_step 16 \
  --checkpoint save/dvs128-gesture_T16/checkpoint_epoch<N>
```

#### Other Neuromorphic Datasets

```bash
# N-Caltech101
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  --dataset /path/to/ncaltech101 \
  --dataset-name ncaltech101 \
  -b 64 --workers 4 --T 10 --epochs 500

# ASL-DVS
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  --dataset /path/to/asl-dvs \
  --dataset-name asl-dvs \
  -b 64 --workers 4 --T 10 --epochs 500
```

---

## Spiking Neuron Models

Three activation function variants are implemented in `modules/spike_layer.py` (and `dvs-SNN/spiking_layer.py`):

| Model | Description | Use case |
|-------|-------------|----------|
| **LIF** | Standard Leaky Integrate-and-Fire with temporal voltage dynamics | Baseline spiking |
| **MixedLIF** | Hybrid: spike path (first half of channels) + ReLU-like continuous path (second half) | Best for contrastive learning views |
| **LIFt** | Dual LIF paths for both contrastive views | Full spike-based contrastive |

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhouspikeclr,
  title={SpikeCLR: Self-Supervised Contrastive Learning for Visual Representations with Spiking Neural Networks},
  author={Zhou, Chengwei and Datta, Gourav},
  booktitle={Forty-third International Conference on Machine Learning}
}
```


