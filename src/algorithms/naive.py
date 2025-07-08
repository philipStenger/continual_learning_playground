"""
Naive continual learning algorithm - simple fine-tuning without any regularization.
This serves as a baseline that typically suffers from catastrophic forgetting.
"""

import torch
import torch.nn.functional as F
from torch import optim


class NaiveLearning:
    """
    Naive continual learning approach that simply fine-tunes the model on each new task
    without any mechanism to prevent catastrophic forgetting.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize the naive learning algorithm.
        
        Args:
            model: The neural network model to train
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        
    def train_task(self, train_loader, test_loader, epochs=10, lr=0.001):
        """
        Train the model on a new task using standard supervised learning.
        
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data (not used in training)
            epochs: Number of training epochs
            lr: Learning rate for optimizer
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total_samples = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                
                # Standard cross-entropy loss
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += target.size(0)
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                          f'Loss: {loss.item():.6f}, '
                          f'Accuracy: {100. * correct / total_samples:.2f}%')
            
            # Print epoch summary
            avg_loss = total_loss / len(train_loader)
            accuracy = 100. * correct / total_samples
            print(f'Epoch {epoch+1}/{epochs} completed - '
                  f'Average Loss: {avg_loss:.6f}, '
                  f'Training Accuracy: {accuracy:.2f}%')
    
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
    
    def get_model_state(self):
        """
        Get the current model state for saving/loading.
        
        Returns:
            dict: Model state dictionary
        """
        return self.model.state_dict()
    
    def load_model_state(self, state_dict):
        """
        Load a model state.
        
        Args:
            state_dict: Model state dictionary to load
        """
        self.model.load_state_dict(state_dict)
