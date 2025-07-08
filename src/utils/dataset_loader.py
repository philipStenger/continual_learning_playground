"""
Dataset loader utilities for continual learning experiments.
"""

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import List, Tuple, Union


def get_dataset(dataset_name: str, batch_size: int = 128, num_tasks: int = 5) -> Tuple[List[data.DataLoader], List[data.DataLoader]]:
    """
    Get train and test data loaders for continual learning scenarios.
    
    Args:
        dataset_name (str): Name of the dataset ('mnist', 'cifar10', 'permuted_mnist')
        batch_size (int): Batch size for data loaders
        num_tasks (int): Number of tasks for continual learning
    
    Returns:
        Tuple[List[DataLoader], List[DataLoader]]: (train_loaders, test_loaders)
    """
    if dataset_name == 'mnist':
        return get_split_mnist(batch_size, num_tasks)
    elif dataset_name == 'cifar10':
        return get_split_cifar10(batch_size, num_tasks)
    elif dataset_name == 'permuted_mnist':
        return get_permuted_mnist(batch_size, num_tasks)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_split_mnist(batch_size: int = 128, num_tasks: int = 5) -> Tuple[List[data.DataLoader], List[data.DataLoader]]:
    """
    Create split MNIST tasks where each task contains a subset of classes.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_tasks (int): Number of tasks (splits)
    
    Returns:
        Tuple[List[DataLoader], List[DataLoader]]: (train_loaders, test_loaders)
    """
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Split classes across tasks
    classes_per_task = 10 // num_tasks
    train_loaders = []
    test_loaders = []
    
    for task_id in range(num_tasks):
        # Define which classes belong to this task
        start_class = task_id * classes_per_task
        end_class = start_class + classes_per_task
        if task_id == num_tasks - 1:  # Last task gets remaining classes
            end_class = 10
        
        task_classes = list(range(start_class, end_class))
        
        # Filter datasets for current task
        train_task_data = filter_dataset_by_class(train_dataset, task_classes)
        test_task_data = filter_dataset_by_class(test_dataset, task_classes)
        
        # Create data loaders
        train_loader = data.DataLoader(
            train_task_data, batch_size=batch_size, shuffle=True, num_workers=2
        )
        test_loader = data.DataLoader(
            test_task_data, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    return train_loaders, test_loaders


def get_split_cifar10(batch_size: int = 128, num_tasks: int = 5) -> Tuple[List[data.DataLoader], List[data.DataLoader]]:
    """
    Create split CIFAR-10 tasks where each task contains a subset of classes.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_tasks (int): Number of tasks (splits)
    
    Returns:
        Tuple[List[DataLoader], List[DataLoader]]: (train_loaders, test_loaders)
    """
    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Split classes across tasks
    classes_per_task = 10 // num_tasks
    train_loaders = []
    test_loaders = []
    
    for task_id in range(num_tasks):
        # Define which classes belong to this task
        start_class = task_id * classes_per_task
        end_class = start_class + classes_per_task
        if task_id == num_tasks - 1:  # Last task gets remaining classes
            end_class = 10
        
        task_classes = list(range(start_class, end_class))
        
        # Filter datasets for current task
        train_task_data = filter_dataset_by_class(train_dataset, task_classes)
        test_task_data = filter_dataset_by_class(test_dataset, task_classes)
        
        # Create data loaders
        train_loader = data.DataLoader(
            train_task_data, batch_size=batch_size, shuffle=True, num_workers=2
        )
        test_loader = data.DataLoader(
            test_task_data, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    return train_loaders, test_loaders


def get_permuted_mnist(batch_size: int = 128, num_tasks: int = 5) -> Tuple[List[data.DataLoader], List[data.DataLoader]]:
    """
    Create permuted MNIST tasks where each task has the same classes but different pixel permutations.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_tasks (int): Number of tasks (permutations)
    
    Returns:
        Tuple[List[DataLoader], List[DataLoader]]: (train_loaders, test_loaders)
    """
    # Base transformation
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loaders = []
    test_loaders = []
    
    for task_id in range(num_tasks):
        # Create permutation for this task
        if task_id == 0:
            # First task uses original MNIST (no permutation)
            permutation = None
        else:
            # Create a random permutation
            np.random.seed(task_id)  # For reproducibility
            permutation = np.random.permutation(784)
        
        # Create transform with permutation
        if permutation is not None:
            transform = transforms.Compose([
                base_transform,
                PermutationTransform(permutation)
            ])
        else:
            transform = base_transform
        
        # Download MNIST dataset with task-specific transform
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        # Create data loaders
        train_loader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        test_loader = data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    return train_loaders, test_loaders


def filter_dataset_by_class(dataset, target_classes: List[int]):
    """
    Filter a dataset to only include samples from specified classes.
    
    Args:
        dataset: PyTorch dataset
        target_classes: List of class indices to keep
    
    Returns:
        Filtered dataset
    """
    indices = []
    for i, (_, label) in enumerate(dataset):
        if label in target_classes:
            indices.append(i)
    
    return data.Subset(dataset, indices)


class PermutationTransform:
    """Transform that applies a fixed permutation to flattened input."""
    
    def __init__(self, permutation):
        self.permutation = permutation
    
    def __call__(self, x):
        # Flatten, permute, and reshape back
        original_shape = x.shape
        x_flat = x.view(-1)
        x_permuted = x_flat[self.permutation]
        return x_permuted.view(original_shape)
