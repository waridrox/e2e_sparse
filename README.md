# End-to-End Sparse Point Cloud Jet Classification

[![ML4Sci](https://img.shields.io/badge/ML4Sci%20-blue)](https://ml4sci.org/gsoc/2025/proposal_E2E2.html)

A deep learning framework for Quark-Gluon jet classification using sparse point cloud representations. This project implements multiple neural network architectures optimized for high-energy physics jet tagging tasks.

## Project Overview

This repository contains the implementation of end-to-end deep learning models for classifying particle jets as quarks or gluons. The framework processes jet data as point clouds, leveraging both convolutional and attention-based architectures to capture geometric and topological features.

**Project Reference:** [ML4Sci GSoC 2025 - End-to-End Sparse Deep Learning](https://ml4sci.org/gsoc/2025/proposal_E2E2.html)

## Architecture

### Supported Models

#### 1. **Sparse CNN (ResNet-based)**
- 1D Convolutional ResNet architecture optimized for point clouds
- Variants: Small (S), Medium (M), Large (L)
- Supports multiple resolutions: 256, 512, 768, 1024 points
- Features residual connections and batch normalization

#### 2. **Aggregation Transformer**
- Self-attention mechanism for point cloud processing
- Global and local feature aggregation
- Positional encoding for spatial relationships
- Variants optimized for different point cloud sizes

### Model Variants

| Model | Resolution | Parameters |
|-------|-----------|--------------|
| ResNet_PC_256_S | 256 points | Small |
| ResNet_PC_512_S | 512 points | Small |
| ResNet_PC_768_S | 768 points | Small |
| ResNet_PC_768_M | 768 points | Medium |
| ResNet_PC_1024_S | 1024 points | Small |
| ResNet_PC_1024_M | 1024 points | Medium |
| ResNet_PC_1024_L | 1024 points | Large |

## Project Structure

```
e2e_sparse/
├── DataGeneration/
│   └── QuarkGluon/
│       ├── ToPointCloudForm.py    # Convert raw data to point cloud format
│       └── 4Resolutions.sh        # Generate datasets at 4 resolutions
│
├── Supervised/
│   ├── CNN/
│   │   └── model.py               # ResNet-based CNN architectures
│   ├── AggregationTransformer/
│   │   └── model.py               # Transformer-based models
│   ├── trainer.py                 # Single-GPU training script
│   ├── trainer4Node.py            # Multi-node distributed training
│   └── Experiments/
│       └── Scripts/
│           ├── bash/              # Local training scripts
│           ├── slurm-no-resume/   # SLURM scripts without resume
│           └── slurm-preempt-chain/  # SLURM with checkpoint resume
│
└── README.md
```

### Data Preparation

1. **Generate Point Cloud Datasets:**
```bash
cd DataGeneration/QuarkGluon
bash 4Resolutions.sh
```

This will create datasets at 4 different resolutions:
- QG256.h5 (256 points)
- QG512.h5 (512 points)
- QG768.h5 (768 points)
- QG1024.h5 (1024 points)

## Training

### Single GPU Training

```bash
python Supervised/trainer.py \
  --datapath=/path/to/QG1024.h5 \
  --Nepochs=100 \
  --lr=1e-3 \
  --model_variant=ResNet_PC_1024_S \
  --UseWandb=True \
  --wandb_project=quark-gluon \
  --wandb_entity=your-entity \
  --wandb_run_name=resnet_1024_experiment \
  --wandb_key=your-api-key \
  --Checkpoint_dir=/path/to/checkpoints \
  --NAccumSteps=1
```

### Multi-GPU Training

```bash
python Supervised/trainer4Node.py \
  --datapath=/path/to/QG1024.h5 \
  --Nepochs=100 \
  --lr=1e-3 \
  --model_variant=Transformer_PC_1024_S \
  --UseWandb=True \
  --wandb_project=quark-gluon \
  --wandb_entity=your-entity \
  --wandb_run_name=transformer_1024_4gpu \
  --wandb_key=your-api-key \
  --Checkpoint_dir=/path/to/checkpoints
```

### SLURM Cluster Training

For HPC clusters with SLURM:

```bash
# Preempt-chain (with automatic resume on preemption)
cd Supervised/Experiments/Scripts/slurm-preempt-chain
sbatch AggregationTransformer1024.sh <run_id>

# Standard SLURM (no resume)
cd Supervised/Experiments/Scripts/slurm-no-resume
sbatch SparseCNNResnet.sh
```
