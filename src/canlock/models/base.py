"""
Base Model for CAN IDS

Abstract base class for all CAN intrusion detection models.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
import json


class CANIDSModel(nn.Module):
    """
    Abstract base class for CAN bus intrusion detection models.
    
    Provides common functionality for saving/loading checkpoints,
    feature extraction, and model management.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 5,  # normal, dos, fuzzing, spoofing, injection
        window_size: int = 50
    ):
        """
        Initialize base model.
        
        Args:
            input_dim: Number of input features per message
            num_classes: Number of output classes (default 5)
            window_size: Number of sequential messages in a window
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.window_size = window_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Must be implemented by subclasses.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract learned features before classification layer.
        
        Useful for visualization and analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        raise NotImplementedError("Subclasses should implement get_features()")
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        optimizer_state: Dict = None,
        metrics: Dict[str, float] = None,
        **kwargs
    ):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current training epoch
            optimizer_state: Optimizer state dict (optional)
            metrics: Training metrics (optional)
            **kwargs: Additional metadata to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'num_classes': self.num_classes,
                'window_size': self.window_size,
                'model_class': self.__class__.__name__
            },
            'metrics': metrics or {},
            **kwargs
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, path)
        
        # Helper function to convert numpy types to native Python types
        def convert_to_native(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        # Also save config as JSON for easy inspection
        config_path = path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump({
                'epoch': int(epoch),
                'model_config': convert_to_native(checkpoint['model_config']),
                'metrics': convert_to_native(checkpoint.get('metrics', {}))
            }, f, indent=2)
        
        print(f"✓ Checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        device: str = 'cpu',
        load_optimizer: bool = False
    ) -> tuple:
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file
            device: Device to load model to ('cpu' or 'cuda')
            load_optimizer: Whether to return optimizer state
            
        Returns:
            Tuple of (model, checkpoint_dict) or (model, checkpoint_dict, optimizer_state)
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=device)
        
        # Extract model config
        config = checkpoint['model_config']
        
        # Create model instance
        model = cls(
            input_dim=config['input_dim'],
            num_classes=config['num_classes'],
            window_size=config['window_size']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            return model, checkpoint, checkpoint['optimizer_state_dict']
        
        return model, checkpoint
    
    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information summary.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_class': self.__class__.__name__,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'window_size': self.window_size,
            'total_parameters': self.count_parameters(),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
