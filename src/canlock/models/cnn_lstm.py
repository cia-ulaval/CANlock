"""
CNN-LSTM Model for CAN Bus Intrusion Detection

Architecture: CNN (spatial feature extraction) → LSTM (temporal modeling)

This architecture first extracts spatial patterns from CAN message features
using convolutional layers, then captures temporal dependencies using LSTM.
"""

import torch
import torch.nn as nn
from .base import CANIDSModel


class CNNLSTM(CANIDSModel):
    """
    CNN-LSTM hybrid architecture for CAN IDS.
    
    Flow: Input → CNN Layers → LSTM Layers → Fully Connected → Output
    
    The CNN component extracts local patterns from message features,
    and the LSTM component models the temporal sequence of these patterns.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 5,
        window_size: int = 50,
        cnn_channels: list = [64, 128],
        cnn_kernel_sizes: list = [5, 3],
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize CNN-LSTM model.
        
        Args:
            input_dim: Number of features per CAN message
            num_classes: Number of output classes
            window_size: Number of messages in sequence
            cnn_channels: List of output channels for each CNN layer
            cnn_kernel_sizes: List of kernel sizes for each CNN layer
            lstm_hidden_size: Hidden size for LSTM layers
            lstm_num_layers: Number of stacked LSTM layers
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super().__init__(input_dim, num_classes, window_size)
        
        self.cnn_channels = cnn_channels
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dropout_prob = dropout
        self.bidirectional = bidirectional
        
        # CNN Layers for spatial feature extraction
        # Input shape: (batch, window_size, input_dim)
        # Need to permute to (batch, input_dim, window_size) for Conv1d
        
        cnn_layers = []
        in_channels = input_dim
        
        for out_channels, kernel_size in zip(cnn_channels, cnn_kernel_sizes):
            cnn_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2  # Same padding
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate output size after CNN
        # After each MaxPool1d(2), sequence length is halved
        self.cnn_output_length = window_size // (2 ** len(cnn_channels))
        self.cnn_output_channels = cnn_channels[-1]
        
        # LSTM Layers for temporal modeling
        # Input to LSTM: (batch, cnn_output_length, cnn_output_channels)
        
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected layers
        lstm_output_size = lstm_hidden_size * (2 if bidirectional else 1)
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN-LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # CNN expects (batch, channels, sequence_length)
        # Permute from (batch, window_size, input_dim) to (batch, input_dim, window_size)
        x = x.permute(0, 2, 1)
        
        # CNN feature extraction
        x = self.cnn(x)  # (batch, cnn_output_channels, cnn_output_length)
        
        # Permute back for LSTM: (batch, cnn_output_length, cnn_output_channels)
        x = x.permute(0, 2, 1)
        
        # LSTM temporal modeling
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state
        # If bidirectional, concatenate forward and backward final states
        if self.bidirectional:
            # hidden shape: (num_layers * 2, batch, hidden_size)
            # Get last layer's forward and backward hidden states
            forward_hidden = hidden[-2]  # Last layer, forward direction
            backward_hidden = hidden[-1]  # Last layer, backward direction
            last_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            last_hidden = hidden[-1]  # Last layer hidden state
        
        # Fully connected classification
        output = self.fc(last_hidden)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract learned features before final classification.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor after LSTM (before FC layers)
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        
        # LSTM temporal modeling
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Return last hidden state concatenated
        if self.bidirectional:
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            features = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            features = hidden[-1]
        
        return features


# Helper function to create default CNN-LSTM model
def create_cnn_lstm(
    input_dim: int,
    num_classes: int = 5,
    window_size: int = 50,
    model_size: str = 'medium'
) -> CNNLSTM:
    """
    Create CNN-LSTM model with predefined configurations.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        window_size: Sequence window size
        model_size: 'small', 'medium', or 'large'
        
    Returns:
        Initialized CNNLSTM model
    """
    configs = {
        'small': {
            'cnn_channels': [32, 64],
            'cnn_kernel_sizes': [3, 3],
            'lstm_hidden_size': 64,
            'lstm_num_layers': 1,
            'dropout': 0.2
        },
        'medium': {
            'cnn_channels': [64, 128],
            'cnn_kernel_sizes': [5, 3],
            'lstm_hidden_size': 128,
            'lstm_num_layers': 2,
            'dropout': 0.3
        },
        'large': {
            'cnn_channels': [128, 256, 512],
            'cnn_kernel_sizes': [7, 5, 3],
            'lstm_hidden_size': 256,
            'lstm_num_layers': 3,
            'dropout': 0.4
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")
    
    config = configs[model_size]
    
    return CNNLSTM(
        input_dim=input_dim,
        num_classes=num_classes,
        window_size=window_size,
        **config
    )
