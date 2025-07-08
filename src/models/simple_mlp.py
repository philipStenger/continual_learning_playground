"""
Simple Multi-Layer Perceptron model for continual learning experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """
    A simple multi-layer perceptron with configurable architecture.
    
    Args:
        input_size (int): Size of input features
        hidden_size (int): Size of hidden layer
        output_size (int): Number of output classes
        num_hidden_layers (int): Number of hidden layers (default: 2)
        dropout_rate (float): Dropout rate for regularization (default: 0.5)
    """
    
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=2, dropout_rate=0.5):
        super(SimpleMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        
        # Create layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_size)
        """
        # Flatten input if needed (for image data)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.network(x)
    
    def get_features(self, x, layer_idx=-2):
        """
        Extract features from a specific layer.
        
        Args:
            x (torch.Tensor): Input tensor
            layer_idx (int): Index of layer to extract features from (-2 for second-to-last layer)
        
        Returns:
            torch.Tensor: Features from the specified layer
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward through network up to specified layer
        for i, layer in enumerate(self.network):
            x = layer(x)
            if i == len(self.network) + layer_idx:
                return x
        
        return x
    
    def freeze_layers(self, num_layers_to_freeze):
        """
        Freeze the first num_layers_to_freeze layers.
        
        Args:
            num_layers_to_freeze (int): Number of layers to freeze from the beginning
        """
        layer_count = 0
        for module in self.network:
            if isinstance(module, nn.Linear):
                if layer_count < num_layers_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False
                layer_count += 1
    
    def unfreeze_all(self):
        """Unfreeze all layers in the network."""
        for param in self.parameters():
            param.requires_grad = True
