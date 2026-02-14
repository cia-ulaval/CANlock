"""
LSTM-CNN Model for CAN Bus Intrusion Detection

Architecture: LSTM (temporal modeling) → CNN (pattern extraction)

This architecture first captures temporal dependencies using LSTM,
then extracts patterns from the LSTM outputs using convolutional layers.
"""

import torch
import torch.nn as nn
from .base import CANIDSModel


class LSTMCNN(CANIDSModel):
    """
    LSTM-CNN hybrid architecture for CAN IDS.
    
    Flow: Input → LSTM Layers → CNN Layers → Global Pooling → Fully Connected → Output
    
    The LSTM component first models temporal sequences,
    then the CNN component extracts high-level patterns from LSTM outputs.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 5,
        window_size: int = 50,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        cnn_channels: list = [64, 128],
        cnn_kernel_sizes: list = [3, 3],
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM-CNN model.
        
        Args:
            input_dim: Number of features per CAN message
            num_classes: Number of output classes
            window_size: Number of messages in sequence
            lstm_hidden_size: Hidden size for LSTM layers
            lstm_num_layers: Number of stacked LSTM layers
            cnn_channels: List of output channels for each CNN layer
            cnn_kernel_sizes: List of kernel sizes for each CNN layer
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super().__init__(input_dim, num_classes, window_size)
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.cnn_channels = cnn_channels
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.dropout_prob = dropout
        self.bidirectional = bidirectional
        
        # LSTM Layers for temporal modeling
        # Input shape: (batch, window_size, input_dim)
        # Output: (batch, window_size, lstm_hidden_size * 2 if bidirectional)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.lstm_dropout = nn.Dropout(dropout)
        
        # CNN Layers for pattern extraction from LSTM outputs
        # LSTM output size per timestep
        lstm_output_size = lstm_hidden_size * (2 if bidirectional else 1)
        
        cnn_layers = []
        in_channels = lstm_output_size
        
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
        
        # Global average pooling to get fixed-size representation
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final output channels from CNN
        self.cnn_output_channels = cnn_channels[-1]
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_output_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM-CNN.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # LSTM temporal modeling
        # Input: (batch, window_size, input_dim)
        # Output: (batch, window_size, lstm_hidden_size * 2)
        lstm_out, (hidden, cell) = self.lstm(x)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # CNN expects (batch, channels, sequence_length)
        # Permute from (batch, window_size, lstm_output_size) to (batch, lstm_output_size, window_size)
        x = lstm_out.permute(0, 2, 1)
        
        # CNN pattern extraction
        x = self.cnn(x)  # (batch, cnn_output_channels, reduced_length)
        
        # Global average pooling
        x = self.global_avg_pool(x)  # (batch, cnn_output_channels, 1)
        x = x.squeeze(-1)  # (batch, cnn_output_channels)
        
        # Fully connected classification
        output = self.fc(x)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract learned features before final classification.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor after CNN global pooling (before FC layers)
        """
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # CNN pattern extraction
        x = lstm_out.permute(0, 2, 1)
        x = self.cnn(x)
        
        # Global average pooling
        features = self.global_avg_pool(x).squeeze(-1)
        
        return features


# Helper function to create default LSTM-CNN model
def create_lstm_cnn(
    input_dim: int,
    num_classes: int = 5,
    window_size: int = 50,
    model_size: str = 'medium'
) -> LSTMCNN:
    """
    Create LSTM-CNN model with predefined configurations.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        window_size: Sequence window size
        model_size: 'small', 'medium', or 'large'
        
    Returns:
        Initialized LSTMCNN model
    """
    configs = {
        'small': {
            'lstm_hidden_size': 64,
            'lstm_num_layers': 1,
            'cnn_channels': [32, 64],
            'cnn_kernel_sizes': [3, 3],
            'dropout': 0.2
        },
        'medium': {
            'lstm_hidden_size': 128,
            'lstm_num_layers': 2,
            'cnn_channels': [64, 128],
            'cnn_kernel_sizes': [3, 3],
            'dropout': 0.3
        },
        'large': {
            'lstm_hidden_size': 256,
            'lstm_num_layers': 3,
            'cnn_channels': [128, 256],
            'cnn_kernel_sizes': [5, 3],
            'dropout': 0.4
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")
    
    config = configs[model_size]
    
    return LSTMCNN(
        input_dim=input_dim,
        num_classes=num_classes,
        window_size=window_size,
        **config
    )
