"""
Experience Replay algorithm for continual learning.
Stores a subset of examples from previous tasks and replays them during training on new tasks.
"""

import torch
import torch.nn.functional as F
from torch import optim
import random
import numpy as np
from collections import defaultdict


class ExperienceReplay:
    """
    Experience Replay continual learning algorithm.
    Maintains a memory buffer of past examples and replays them during training.
    """
    
    def __init__(self, model, device='cpu', memory_size=1000, memory_per_task=None):
        """
        Initialize the Experience Replay algorithm.
        
        Args:
            model: The neural network model to train
            device: Device to run computations on ('cpu' or 'cuda')
            memory_size: Total size of the memory buffer
            memory_per_task: Memory size per task (if None, uses equal allocation)
        """
        self.model = model
        self.device = device
        self.memory_size = memory_size
        self.memory_per_task = memory_per_task
        
        # Memory buffer to store past examples
        self.memory_data = []
        self.memory_targets = []
        self.memory_task_ids = []
        
        # Track seen tasks
        self.seen_tasks = 0
        
    def train_task(self, train_loader, test_loader, epochs=10, lr=0.001, replay_ratio=0.5):
        """
        Train the model on a new task with experience replay.
        
        Args:
            train_loader: DataLoader for current task training data
            test_loader: DataLoader for test data (not used in training)
            epochs: Number of training epochs
            lr: Learning rate for optimizer
            replay_ratio: Ratio of replay samples to current task samples in each batch
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Store examples from current task in memory
        self._update_memory(train_loader, self.seen_tasks)
        self.seen_tasks += 1
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            current_loss = 0
            replay_loss = 0
            correct = 0
            total_samples = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.size(0)
                
                # Calculate replay batch size
                replay_batch_size = int(batch_size * replay_ratio)
                
                optimizer.zero_grad()
                
                # Current task loss
                output_current = self.model(data)
                loss_current = F.cross_entropy(output_current, target)
                current_loss += loss_current.item()
                
                total_loss_batch = loss_current
                loss_replay_item = 0  # Initialize replay loss
                
                # Replay loss (if we have memory)
                if len(self.memory_data) > 0 and replay_batch_size > 0:
                    replay_data, replay_targets = self._sample_memory(replay_batch_size)
                    replay_data = replay_data.to(self.device)
                    replay_targets = replay_targets.to(self.device)
                    
                    output_replay = self.model(replay_data)
                    loss_replay_tensor = F.cross_entropy(output_replay, replay_targets)
                    loss_replay_item = loss_replay_tensor.item()
                    replay_loss += loss_replay_item
                    
                    total_loss_batch = total_loss_batch + loss_replay_tensor
                
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                
                # Calculate accuracy on current task
                pred = output_current.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += target.size(0)
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                          f'Total Loss: {total_loss_batch.item():.6f}, '
                          f'Current Loss: {loss_current.item():.6f}, '
                          f'Replay Loss: {loss_replay_item:.6f}')
            
            # Print epoch summary
            avg_loss = total_loss / len(train_loader)
            avg_current_loss = current_loss / len(train_loader)
            avg_replay_loss = replay_loss / len(train_loader) if len(self.memory_data) > 0 else 0
            accuracy = 100. * correct / total_samples
            
            print(f'Epoch {epoch+1}/{epochs} completed - '
                  f'Total Loss: {avg_loss:.6f}, '
                  f'Current Loss: {avg_current_loss:.6f}, '
                  f'Replay Loss: {avg_replay_loss:.6f}, '
                  f'Training Accuracy: {accuracy:.2f}%')
    
    def _update_memory(self, train_loader, task_id):
        """
        Update memory buffer with examples from the current task.
        
        Args:
            train_loader: DataLoader for current task
            task_id: ID of the current task
        """
        # Determine memory allocation per task
        if self.memory_per_task is not None:
            memory_per_task = self.memory_per_task
        else:
            memory_per_task = self.memory_size // (self.seen_tasks + 1)
        
        # Collect all examples from current task
        task_data = []
        task_targets = []
        
        for data, target in train_loader:
            for i in range(data.size(0)):
                task_data.append(data[i].cpu())
                task_targets.append(target[i].cpu())
        
        # Randomly sample examples for memory
        if len(task_data) > memory_per_task:
            indices = random.sample(range(len(task_data)), memory_per_task)
            task_data = [task_data[i] for i in indices]
            task_targets = [task_targets[i] for i in indices]
        
        # Add to memory
        for data, target in zip(task_data, task_targets):
            self.memory_data.append(data)
            self.memory_targets.append(target)
            self.memory_task_ids.append(task_id)
        
        # If memory is full, remove oldest examples
        if len(self.memory_data) > self.memory_size:
            # Remove excess examples (FIFO)
            excess = len(self.memory_data) - self.memory_size
            self.memory_data = self.memory_data[excess:]
            self.memory_targets = self.memory_targets[excess:]
            self.memory_task_ids = self.memory_task_ids[excess:]
        
        print(f"Memory updated: {len(self.memory_data)} examples stored")
    
    def _sample_memory(self, batch_size):
        """
        Sample a batch of examples from memory.
        
        Args:
            batch_size: Number of examples to sample
            
        Returns:
            Tuple of (data_batch, targets_batch)
        """
        if len(self.memory_data) == 0:
            return torch.tensor([]), torch.tensor([])
        
        # Sample random indices
        indices = random.sample(range(len(self.memory_data)), 
                               min(batch_size, len(self.memory_data)))
        
        # Get sampled data
        sampled_data = torch.stack([self.memory_data[i] for i in indices])
        sampled_targets = torch.stack([self.memory_targets[i] for i in indices])
        
        return sampled_data, sampled_targets
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            float: Accuracy on the test set
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / total
    
    def get_memory_stats(self):
        """
        Get statistics about the memory buffer.
        
        Returns:
            dict: Memory statistics
        """
        if len(self.memory_task_ids) == 0:
            return {"total_examples": 0, "tasks": {}}
        
        task_counts = defaultdict(int)
        for task_id in self.memory_task_ids:
            task_counts[task_id] += 1
        
        return {
            "total_examples": len(self.memory_data),
            "tasks": dict(task_counts),
            "memory_utilization": len(self.memory_data) / self.memory_size
        }
    
    def clear_memory(self):
        """Clear the memory buffer."""
        self.memory_data = []
        self.memory_targets = []
        self.memory_task_ids = []
        self.seen_tasks = 0
