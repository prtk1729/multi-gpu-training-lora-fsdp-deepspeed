# multi-gpu-training-lora-fsdp-deepspeed
Fine-tuning large models using Multi-GPU techniques including `DDP, FSDP, Model Parallelism, DeepSpeed, and LoRA` with PyTorch and Hugging Face Accelerate.

## Distributed Fine-Tuning of Large Language Models

This repository provides a comprehensive exploration and practical implementation of techniques for efficiently fine-tuning Large Language Models (LLMs) on multi-GPU systems. As LLMs grow in size, single-GPU training becomes infeasible or prohibitively slow. This project addresses these challenges by demonstrating various distributed training paradigms and VRAM optimization strategies.

[## Table of Contents

- [Introduction](#introduction)
- [Key Areas Explored](#key-areas-explored)
  - [VRAM Management](#vram-management)
  - [Distributed Training Strategies](#distributed-training-strategies)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Performance Analysis](#performance-analysis)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Fine-tuning LLMs is a crucial step for adapting pre-trained models to specific tasks or datasets. However, the sheer size of modern LLMs (billions of parameters) demands sophisticated approaches for efficient training, especially when leveraging multiple GPUs. This repository aims to provide a clear understanding and practical examples of how to achieve this, focusing on optimizing for both training speed and VRAM efficiency.

## Key Areas Explored

### VRAM Management

Understanding and optimizing VRAM usage is paramount in LLM training. This project explores:

* **Detailed VRAM Requirements Calculation**: Analysis of how model parameters, gradients, optimizer states (especially for Adam), and activations contribute to VRAM consumption.
* **Impact of Data Types**: Examination of how precision (e.g., 16-bit vs. 32-bit floats) influences memory footprint.
* **LoRA (Low-Rank Adaptation)**: Implementation and analysis of LoRA for significantly reducing the number of trainable parameters and VRAM requirements by freezing the base model and training small, adaptive matrices.
* **Quantization**: Exploring how techniques like 4-bit quantization can further reduce model size and memory footprint, especially when combined with LoRA (QLoRA).
* **Batch Size and Context Length**: Demonstrating their direct correlation with activation memory and overall VRAM usage.

### Distributed Training Strategies

This repository implements and compares various multi-GPU training approaches:

* **Naive Model Parallelism**: Distributing model layers across different GPUs when a model cannot fit on a single device. This highlights the concept of pipelining forward and backward passes.
* **Distributed Data Parallel (DDP)**: Replicating the full model on each GPU and distributing batches of data. This strategy is ideal for models that fit on a single GPU and aims to accelerate training by processing data in parallel.
* **Fully Sharded Data Parallel (FSDP)**: An advanced technique where model parameters, gradients, and optimizer states are sharded across GPUs. This enables training of extremely large models that would otherwise not fit on any single GPU, by dynamically re-assembling necessary components during forward and backward passes.
* **Communication Overhead Analysis**: Discussion and implicit demonstration of the trade-offs in communication between GPUs for each parallelization strategy.

## Project Structure
```text
├── src/
│   ├── models/              # LLM architectures (e.g., TinyLlama, CodeLlama adaptions)
│   ├── training/            # Core training loops and utilities
│   │   ├── model_parallel.py   # Implementation for Naive Model Parallelism
│   │   ├── ddp_training.py     # Implementation for Distributed Data Parallel
│   │   ├── fsdp_training.py    # Implementation for Fully Sharded Data Parallel
│   │   └── optimizers.py       # Custom optimizer configurations and Adam details
│   ├── utils/               # Utility functions (VRAM calculation, data loading, etc.)
│   └── main.py              # Entry point for running training experiments
├── configs/                 # Configuration files for different models and training setups
├── data/                    # Sample datasets for fine-tuning
├── notebooks/               # Jupyter notebooks for VRAM analysis and smaller experiments
├── results/                 # Directory to store training logs and performance metrics
└── README.md                # This file
```

## Getting Started

### Prerequisites

* Python 3.9+
* PyTorch (with GPU support)
* Transformers library
* Accelerate library (for DDP/FSDP)
* NVIDIA GPUs (recommended for multi-GPU training)

### Installation

```bash
git clone [https://github.com/your-username/distributed-llm-finetuning.git](https://github.com/your-username/distributed-llm-finetuning.git)
cd distributed-llm-finetuning
pip install -r requirements.txt
```

## Usage

To run a distributed fine-tuning experiment, you'll typically use main.py with a configuration file.
Example for Distributed Data Parallel training with TinyLlama:

```bash
accelerate launch --num_processes=2 src/main.py --config_path configs/tinyllama_ddp.yaml
```
Example for FSDP training with CodeLlama:
```bash
accelerate launch --num_processes=4 src/main.py --config_path configs/codellama_fsdp.yaml
```](https://github.com/prtk1729/multi-gpu-training-lora-fsdp-deepspeed/tree/main)
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
