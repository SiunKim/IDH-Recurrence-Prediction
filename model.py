"""
LSTM-based classifier for IDH prediction.

This module implements a hybrid neural network that combines:
- LSTM layers for processing time series features
- Fully connected layers for time-invariant features
- Embedding generation for downstream tasks

The model architecture:
1. Time-invariant branch: FC -> ReLU
2. Time series branch: LSTM
3. Concatenation of both branches
4. Final classification layers: FC -> ReLU -> FC
"""

from typing import Tuple, List

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Hybrid LSTM classifier combining time series and time-invariant features.
    
    Args:
        input_dim_time_inv: Input dimension of time-invariant features
        input_dim_time_vars: Input dimension of time series features
        hidden_dim_time_inv: Hidden dimension for time-invariant branch
        hidden_dim_time_vars: Hidden dimension for LSTM
        embedding_dim: Dimension of final embedding layer
        layer_dim: Number of LSTM layers
        output_dim: Output dimension (number of classes)
    
    Architecture:
        Time-invariant branch:
            input -> FC -> ReLU
        
        Time series branch:
            input -> LSTM -> last hidden state
        
        Combined:
            concatenate -> FC -> ReLU -> FC
    """
    
    def __init__(self,
                 input_dim_time_inv: int,
                 input_dim_time_vars: int,
                 hidden_dim_time_inv: int,
                 hidden_dim_time_vars: int,
                 embedding_dim: int,
                 layer_dim: int,
                 output_dim: int):
        """Initialize model architecture."""
        super().__init__()
        
        # Save dimensions
        self.input_dim_time_inv = input_dim_time_inv
        self.input_dim_time_vars = input_dim_time_vars
        self.hidden_dim_time_inv = hidden_dim_time_inv
        self.hidden_dim_time_vars = hidden_dim_time_vars
        self.embedding_dim = embedding_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        
        # Time-invariant branch
        self.fc_time_inv = nn.Linear(
            input_dim_time_inv,
            hidden_dim_time_inv
        )
        
        # Time series branch
        self.lstm = nn.LSTM(
            input_size=input_dim_time_vars,
            hidden_size=hidden_dim_time_vars,
            num_layers=layer_dim,
            batch_first=True
        )
        
        # Combined layers
        self.fc1 = nn.Linear(
            hidden_dim_time_inv + hidden_dim_time_vars,
            embedding_dim
        )
        self.fc2 = nn.Linear(
            embedding_dim,
            output_dim
        )
        
        self.relu = nn.ReLU()

    def forward(self, 
                x_time_inv: torch.Tensor,
                x_time_vars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x_time_inv: Time-invariant features [batch_size, input_dim_time_inv]
            x_time_vars: Time series features [batch_size, seq_len, input_dim_time_vars]
            
        Returns:
            tuple of:
                - Model output (logits) [batch_size, output_dim]
                - Embedding vectors [batch_size, embedding_dim]
        """
        # Process time series branch
        h0, c0 = self.init_hidden(x_time_vars)
        out_time_vars, (_, _) = self.lstm(x_time_vars, (h0, c0))
        out_time_vars_last = out_time_vars[:, -1, :]  # Get last hidden state
        
        # Process time-invariant branch
        out_time_inv = self.relu(self.fc_time_inv(x_time_inv))
        
        # Combine branches
        combined = torch.cat((out_time_inv, out_time_vars_last), dim=1)
        embedding = self.relu(self.fc1(combined))
        output = self.fc2(embedding)
        
        return output, embedding

    def init_hidden(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Initialize hidden states for LSTM.
        
        Args:
            x: Input tensor to get batch size
            
        Returns:
            List of initial hidden state and cell state tensors
        """
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim_time_vars)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim_time_vars)
        
        # Move to GPU if available
        return [t.cuda() for t in (h0, c0)]


def create_model(config: dict) -> LSTMClassifier:
    """
    Create LSTM classifier model from configuration.
    
    Args:
        config: Dictionary containing model configuration
            Required keys:
            - input_dim_time_inv
            - input_dim_time_vars  
            - hidden_dim_time_inv
            - hidden_dim_time_vars
            - embedding_dim
            - layer_dim
            - output_dim
            
    Returns:
        Initialized LSTMClassifier model
        
    Example:
        >>> config = {
        ...     'input_dim_time_inv': 20,
        ...     'input_dim_time_vars': 10,
        ...     'hidden_dim_time_inv': 64,
        ...     'hidden_dim_time_vars': 64,
        ...     'embedding_dim': 32,
        ...     'layer_dim': 2,
        ...     'output_dim': 2
        ... }
        >>> model = create_model(config)
    """
    return LSTMClassifier(
        input_dim_time_inv=config['input_dim_time_inv'],
        input_dim_time_vars=config['input_dim_time_vars'],
        hidden_dim_time_inv=config['hidden_dim_time_inv'], 
        hidden_dim_time_vars=config['hidden_dim_time_vars'],
        embedding_dim=config['embedding_dim'],
        layer_dim=config['layer_dim'],
        output_dim=config['output_dim']
    )
