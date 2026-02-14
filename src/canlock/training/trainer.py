"""
IDS Trainer

Training loop for CAN intrusion detection models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from typing import Dict, Optional, Callable
from tqdm import tqdm
import numpy as np
import time

from canlock.models.base import CANIDSModel
from canlock.training.metrics import calculate_metrics, print_metrics


class IDSTrainer:
    """
    Training pipeline for CAN IDS models.
    
    Handles training loop, validation, early stopping, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: CANIDSModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        class_weights: Optional[torch.Tensor] = None,
        checkpoint_dir: Path = Path("checkpoints"),
        early_stopping_patience: int = 10
    ):
        """
        Initialize trainer.
        
        Args:
            model: CANIDSModel instance
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on ('cpu' or 'cuda')
            lr: Learning rate
            weight_decay: L2 regularization
            class_weights: Class weights for imbalanced data (optional)
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Loss function
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self) -> tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> tuple[float, float, dict]:
        """
        Validate model.
        
        Returns:
            Tuple of (average_loss, accuracy, metrics_dict)
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc="Validation")
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            total_loss += loss.item()
            
            # Predictions
            _, predicted = output.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        metrics = calculate_metrics(all_targets, all_predictions)
        
        return avg_loss, metrics['accuracy'], metrics
    
    def train(
        self,
        num_epochs: int,
        verbose: bool = True,
        save_best: bool = True
    ) -> Dict:
        """
        Train model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            verbose: Print progress
            save_best: Save best model checkpoint
            
        Returns:
            Training history dictionary
        """
        print(f"\nStarting training on {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n[Epoch {epoch}/{num_epochs}]")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch summary
            if verbose:
                print(f"\nEpoch Summary:")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
                print(f"  Val F1 (macro): {val_metrics['f1_macro']:.4f}")
                print(f"  Learning Rate: {current_lr:.6f}")
            
            # Early stopping and checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                
                if save_best:
                    self.save_checkpoint(
                        epoch,
                        f"best_model_{self.model.__class__.__name__.lower()}.pt",
                        metrics=val_metrics
                    )
                    print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epoch(s)")
                
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Total time: {elapsed_time/60:.2f} minutes")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return self.history
    
    def save_checkpoint(
        self,
        epoch: int,
        filename: str,
        metrics: Optional[Dict] = None
    ):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            filename: Checkpoint filename
            metrics: Validation metrics to save
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        self.model.save_checkpoint(
            checkpoint_path,
            epoch=epoch,
            optimizer_state=self.optimizer.state_dict(),
            metrics=metrics,
            history=self.history
        )
    
    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> Dict:
        """
        Test model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with test metrics
        """
        print("\nEvaluating on test set...")
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probas = []
        
        for data, target in tqdm(test_loader, desc="Testing"):
            data = data.to(self.device)
            
            # Forward pass
            output = self.model(data)
            probas = torch.softmax(output, dim=1)
            
            # Predictions
            _, predicted = output.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())
            all_probas.extend(probas.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probas = np.array(all_probas)
        
        metrics = calculate_metrics(all_targets, all_predictions, all_probas)
        
        print_metrics(metrics)
        
        return metrics


def calculate_class_weights(train_labels: np.ndarray, device: str = 'cpu') -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        train_labels: Training labels array
        device: Device to place weights on
        
    Returns:
        Tensor of class weights
    """
    unique, counts = np.unique(train_labels, return_counts=True)
    total = len(train_labels)
    
    # Inverse frequency weighting  
    weights = total / (len(unique) * counts)
    
    # Convert to tensor
    weight_tensor = torch.zeros(len(unique))
    for idx, weight in zip(unique, weights):
        weight_tensor[idx] = weight
    
    print(f"Class weights: {weight_tensor.numpy()}")
    
    return weight_tensor.to(device)
