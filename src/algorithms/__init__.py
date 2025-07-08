"""
Elastic Weight Consolidation (EWC) implementation for continual learning.
"""

import torch
import torch.nn.functional as F
from torch import optim
import copy

class EWC:
    """Elastic Weight Consolidation for continual learning."""
    
    def __init__(self, model, device='cpu', lambda_ewc=400):
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        
        # Store parameters and Fisher Information for previous tasks
        self.params_old = {}
        self.fisher_matrix = {}
        
    def train_task(self, train_loader, test_loader, epochs=10, lr=0.001):
        """Train on a new task."""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                
                # Standard cross-entropy loss
                ce_loss = F.cross_entropy(output, target)
                
                # EWC penalty term
                ewc_loss = self._ewc_penalty()
                
                total_loss = ce_loss + ewc_loss
                total_loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.6f}')
        
        # After training, compute Fisher Information Matrix
        self._compute_fisher_matrix(train_loader)
        self._store_old_params()
        
    def _ewc_penalty(self):
        """Compute EWC penalty term."""
        if not self.params_old:
            return torch.tensor(0.0).to(self.device)
        
        penalty = 0
        for name, param in self.model.named_parameters():
            if name in self.params_old:
                penalty += torch.sum(self.fisher_matrix[name] * 
                                   (param - self.params_old[name]).pow(2))
        
        return self.lambda_ewc * penalty
    
    def _compute_fisher_matrix(self, data_loader):
        """Compute Fisher Information Matrix."""
        fisher = {}
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        self.model.eval()
        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            self.model.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) / len(data_loader)
        
        self.fisher_matrix = fisher
    
    def _store_old_params(self):
        """Store current parameters."""
        self.params_old = {}
        for name, param in self.model.named_parameters():
            self.params_old[name] = param.data.clone()
    
    def evaluate(self, test_loader):
        """Evaluate model on test data."""
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
