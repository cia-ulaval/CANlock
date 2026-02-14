"""
Evaluation Metrics for CAN IDS

Functions for calculating and visualizing performance metrics.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Tuple


# Class labels
CLASS_NAMES = ['Normal', 'DoS', 'Fuzzing', 'Spoofing', 'Injection']


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (optional, for ROC-AUC)
        
    Returns:
        Dictionary with metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro and micro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
    }
    
    # Per-class metrics
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(precision):
            metrics[f'precision_{class_name.lower()}'] = precision[i]
            metrics[f'recall_{class_name.lower()}'] = recall[i]
            metrics[f'f1_{class_name.lower()}'] = f1[i]
            metrics[f'support_{class_name.lower()}'] = support[i]
    
    # ROC-AUC if probabilities provided
    if y_pred_proba is not None:
        try:
            # Multi-class ROC-AUC (one-vs-rest)
            roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
            metrics['roc_auc_macro'] = roc_auc
        except ValueError:
            # If some classes missing in test set
            pass
    
    return metrics


def print_metrics(metrics: Dict[str, float]):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics()
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print(f"\n{'Overall Performance':^60}")
    print("-"*60)
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):    {metrics['f1_macro']:.4f}")
    print(f"  F1 (micro):    {metrics['f1_micro']:.4f}")
    if 'roc_auc_macro' in metrics:
        print(f"  ROC-AUC:       {metrics['roc_auc_macro']:.4f}")
    
    print(f"\n{'Per-Class Performance':^60}")
    print("-"*60)
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-"*60)
    
    for class_name in CLASS_NAMES:
        class_key = class_name.lower()
        if f'precision_{class_key}' in metrics:
            precision = metrics[f'precision_{class_key}']
            recall = metrics[f'recall_{class_key}']
            f1 = metrics[f'f1_{class_key}']
            support = int(metrics[f'support_{class_key}'])
            print(f"{class_name:<15} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")
    
    print("="*60 + "\n")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
    title: str = "Confusion Matrix"
) -> go.Figure:
    """
    Plot confusion matrix using plotly.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: If True, normalize by true label
        title: Plot title
        
    Returns:
        Plotly figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        text_template = '%{z:.2%}'
    else:
        text_template = '%{z}'
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=CLASS_NAMES[:len(np.unique(y_true))],
        y=CLASS_NAMES[:len(np.unique(y_true))],
        colorscale='Blues',
        text=cm,
        texttemplate=text_template,
        textfont={"size": 12},
        colorbar=dict(title="Proportion" if normalize else "Count")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        width=600,
        height=500,
        xaxis={'side': 'bottom'},
    )
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History"
) -> go.Figure:
    """
    Plot training and validation loss/accuracy over epochs.
    
    Args:
        history: Dictionary with keys ['train_loss', 'val_loss', 'train_acc', 'val_acc']
        title: Plot title
        
    Returns:
        Plotly figure with subplots
    """
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Loss', 'Accuracy'),
        horizontal_spacing=0.15
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss', 
                   mode='lines+markers', line=dict(color='#3498db')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss', 
                   mode='lines+markers', line=dict(color='#e74c3c')),
        row=1, col=1
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_acc'], name='Train Acc', 
                   mode='lines+markers', line=dict(color='#3498db'), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_acc'], name='Val Acc', 
                   mode='lines+markers', line=dict(color='#e74c3c'), showlegend=False),
        row=1, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    
    fig.update_layout(
        title_text=title,
        height=400,
        width=1000,
        showlegend=True
    )
    
    return fig


def plot_class_distribution(y: np.ndarray, title: str = "Class Distribution") -> go.Figure:
    """
    Plot distribution of classes.
    
    Args:
        y: Labels array
        title: Plot title
        
    Returns:
        Plotly bar chart
    """
    unique, counts = np.unique(y, return_counts=True)
    class_labels = [CLASS_NAMES[i] for i in unique]
    
    fig = go.Figure(data=[
        go.Bar(x=class_labels, y=counts, marker_color='#3498db')
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Class",
        yaxis_title="Count",
        height=400,
        width=600
    )
    
    return fig


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Get classification report as string.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Classification report string
    """
    target_names = [CLASS_NAMES[i] for i in range(len(np.unique(np.concatenate([y_true, y_pred]))))]
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
