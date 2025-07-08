"""
Continual Learning Playground

Main entry point for running continual learning experiments.
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from src.models.simple_mlp import SimpleMLP
from src.utils.dataset_loader import get_dataset
from src.algorithms.ewc import EWC
from src.algorithms.replay import ExperienceReplay
from src.algorithms.naive import NaiveLearning
from src.utils.logging import setup_logger

def main():
    parser = argparse.ArgumentParser(description='Continual Learning Playground')
    parser.add_argument('--algorithm', type=str, default='ewc', 
                       choices=['ewc', 'replay', 'naive'],
                       help='Continual learning algorithm to use')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10', 'permuted_mnist'],
                       help='Dataset to use for experiments')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs per task')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--memory_size', type=int, default=1000,
                       help='Memory buffer size for replay algorithm')
    parser.add_argument('--replay_ratio', type=float, default=0.5,
                       help='Ratio of replay samples to current task samples')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger('continual_learning')
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    train_loaders, test_loaders = get_dataset(args.dataset)
    
    # Initialize model
    model = SimpleMLP(input_size=784, hidden_size=400, output_size=10)
    model.to(device)
    
    # Initialize continual learning algorithm
    device_str = str(device)
    if args.algorithm == 'ewc':
        algorithm = EWC(model, device=device_str)
    elif args.algorithm == 'replay':
        algorithm = ExperienceReplay(model, device=device_str, memory_size=args.memory_size)
    elif args.algorithm == 'naive':
        algorithm = NaiveLearning(model, device=device_str)
    else:
        raise NotImplementedError(f"Algorithm {args.algorithm} not implemented yet")
    
    # Train on tasks sequentially
    for task_id, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
        logger.info(f"Training on task {task_id + 1}")
        
        # Pass additional arguments for replay algorithm
        if args.algorithm == 'replay':
            # Use ExperienceReplay specific method call
            algorithm.train_task(train_loader, test_loader, epochs=args.epochs,  # type: ignore
                               lr=args.lr, replay_ratio=args.replay_ratio)
        else:
            algorithm.train_task(train_loader, test_loader, epochs=args.epochs, lr=args.lr)
        
        # Evaluate on all previous tasks
        logger.info(f"Evaluating after task {task_id + 1}")
        for prev_task_id, prev_test_loader in enumerate(test_loaders[:task_id + 1]):
            accuracy = algorithm.evaluate(prev_test_loader)
            logger.info(f"Task {prev_task_id + 1} accuracy: {accuracy:.4f}")
            
        # Print memory stats for replay algorithm
        if args.algorithm == 'replay' and hasattr(algorithm, 'get_memory_stats'):
            memory_stats = algorithm.get_memory_stats()  # type: ignore
            logger.info(f"Memory stats: {memory_stats}")

if __name__ == "__main__":
    main()
