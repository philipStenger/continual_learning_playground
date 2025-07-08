# Continual Learning Playground

A comprehensive framework for experimenting with continual learning algorithms and comparing their effectiveness in mitigating catastrophic forgetting.

## Overview

This project implements and compares three fundamental continual learning approaches:

### 🧠 **Implemented Algorithms**
- **Naive Learning**: Baseline fine-tuning approach (demonstrates catastrophic forgetting)
- **Elastic Weight Consolidation (EWC)**: Regularization-based method using Fisher Information Matrix
- **Experience Replay**: Memory-based approach with episodic buffer storage

### 🎯 **Methodology**

The framework evaluates continual learning algorithms using sequential task learning scenarios:

1. **Task Sequence**: Models learn multiple tasks sequentially without access to previous task data
2. **Evaluation Protocol**: After each task, performance is measured on all previously learned tasks
3. **Catastrophic Forgetting Assessment**: Quantifies knowledge retention across the task sequence

### 📊 **Supported Scenarios**
- **Split MNIST/CIFAR-10**: Different classes assigned to different tasks
- **Permuted MNIST**: Same classes with different pixel permutations per task
- **Configurable Task Sequences**: Flexible number of tasks and epochs per task

## 🚀 **Quick Start**

### **Minimal Setup (3 commands)**
```bash
# 1. Create and activate virtual environment
python -m venv env && env\Scripts\activate  # Windows
# python -m venv env && source env/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run your first experiment (datasets auto-download!)
python main.py --algorithm ewc --dataset mnist --epochs 5
```

### **First Run Expectations**
- **First time**: ~1-2 minutes (includes dataset download)
- **Subsequent runs**: ~30 seconds to start
- **Dataset storage**: ~200MB total for all datasets

## Setup

1. Create a virtual environment:
```bash
python -m venv env
```

2. Activate the virtual environment:
```bash
# Windows
env\Scripts\activate
# Linux/Mac
source env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python main.py --algorithm [ewc|replay|naive] --dataset [mnist|cifar10|permuted_mnist] --epochs 10
```

### Algorithm-Specific Examples

**EWC (Elastic Weight Consolidation)**
```bash
python main.py --algorithm ewc --dataset mnist --epochs 10 --lr 0.001
```

**Experience Replay**
```bash
python main.py --algorithm replay --dataset permuted_mnist --memory_size 1000 --replay_ratio 0.5 --epochs 10
```

**Naive Baseline**
```bash
python main.py --algorithm naive --dataset cifar10 --epochs 5 --lr 0.001
```

### Command Line Arguments
- `--algorithm`: Continual learning algorithm (`ewc`, `replay`, `naive`)
- `--dataset`: Dataset scenario (`mnist`, `cifar10`, `permuted_mnist`)
- `--epochs`: Number of training epochs per task
- `--lr`: Learning rate for optimization
- `--memory_size`: Memory buffer size for replay algorithm
- `--replay_ratio`: Proportion of replay samples vs. current task samples

### Testing Setup
```bash
python test_setup.py
```

## Project Structure

```
continual_learning_playground/
├── main.py                    # Main entry point for experiments
├── test_setup.py             # Setup verification script
├── requirements.txt          # Python dependencies
├── src/
│   ├── algorithms/
│   │   ├── ewc.py           # Elastic Weight Consolidation
│   │   ├── replay.py        # Experience Replay
│   │   └── naive.py         # Naive baseline
│   ├── models/
│   │   └── simple_mlp.py    # Multi-layer perceptron model
│   ├── data/
│   │   └── dataset_loader.py # Dataset handling utilities
│   └── utils/
│       └── logging.py       # Logging configuration
└── env/                     # Virtual environment
```

## Algorithm Details

### 1. **Naive Learning**
- **Method**: Standard fine-tuning without continual learning mechanisms
- **Pros**: Simple, computationally efficient
- **Cons**: Severe catastrophic forgetting
- **Use Case**: Baseline for comparison

### 2. **Elastic Weight Consolidation (EWC)**
- **Method**: Regularization using Fisher Information Matrix to preserve important weights
- **Pros**: No additional memory for old data, theoretically grounded
- **Cons**: Approximation quality depends on Fisher Information estimation
- **Key Parameter**: `lambda_ewc` (regularization strength)

### 3. **Experience Replay**
- **Method**: Stores subset of previous task examples in memory buffer
- **Pros**: Direct access to old data, strong empirical performance
- **Cons**: Memory overhead, potential privacy concerns
- **Key Parameters**: `memory_size`, `replay_ratio`

## 📦 **Dataset Information**

### **Automatic Dataset Downloads**
The project uses **PyTorch's torchvision** for automatic dataset handling. **No manual dataset preparation required!**

When you run the project for the first time with any dataset, it will automatically:
1. Create a `./data/` directory in your project folder
2. Download the required dataset from official sources
3. Cache the dataset locally for future runs

### **Supported Datasets & Formats**

#### **1. MNIST**
- **Source**: Automatic download via `torchvision.datasets.MNIST`
- **Format**: 28x28 grayscale images, 10 classes (digits 0-9)
- **Size**: ~12MB download
- **Usage**: `--dataset mnist`
- **Tasks**: Classes split across 5 tasks (2 classes per task by default)

#### **2. CIFAR-10**
- **Source**: Automatic download via `torchvision.datasets.CIFAR10`
- **Format**: 32x32 RGB images, 10 classes
- **Size**: ~170MB download
- **Usage**: `--dataset cifar10`
- **Tasks**: Classes split across 5 tasks (2 classes per task by default)

#### **3. Permuted MNIST**
- **Source**: Based on MNIST with pixel permutations
- **Format**: 28x28 grayscale images with different permutation per task
- **Size**: Same as MNIST (~12MB)
- **Usage**: `--dataset permuted_mnist`
- **Tasks**: Same 10 classes but different pixel arrangements per task

### **Data Directory Structure**
After first run, your project will look like:
```
continual_learning_playground/
├── data/                    # Automatically created
│   ├── MNIST/
│   │   └── raw/            # Downloaded MNIST files
│   └── CIFAR10/
│       └── raw/            # Downloaded CIFAR-10 files
├── main.py
└── ...
```

### **Data Preprocessing**
All datasets include automatic preprocessing:
- **Normalization**: Dataset-specific mean/std normalization
- **Tensor Conversion**: PIL Images → PyTorch tensors
- **Data Augmentation**: Random crops and flips for CIFAR-10
- **Task Splitting**: Automatic division into continual learning tasks

### **No Internet? No Problem!**
If you've run the project before, datasets are cached locally and don't require re-downloading.

### **Custom Datasets**
To add your own dataset, modify `src/data/dataset_loader.py`:
```python
def get_custom_dataset(batch_size=128, num_tasks=5):
    # Your custom dataset loading logic here
    pass
```
