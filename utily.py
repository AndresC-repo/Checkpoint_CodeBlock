import torch
import json
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt

def load_checkpoint_state(filepath: str, device: str = 'cpu') -> Dict[str, Any]:
    """Utility to load checkpoint state without full manager"""
    return torch.load(filepath, map_location=device)

def get_available_checkpoints(checkpoint_dir: str, model_name: str) -> Dict[str, str]:
    """Get all available checkpoints for a model"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = {}
    
    patterns = [
        f"{model_name}.pt",
        f"{model_name}_latest.pt", 
        f"{model_name}_best_*.pt"
    ]
    
    for pattern in patterns:
        for checkpoint_file in checkpoint_dir.glob(pattern):
            checkpoint_type = checkpoint_file.stem.replace(f"{model_name}_", "")
            checkpoints[checkpoint_type] = str(checkpoint_file)
    
    return checkpoints

def plot_training_history(training_history: List[Dict[str, Any]], 
                         metrics: List[str] = None,
                         save_path: str = None):
    """Plot training history metrics"""
    if not metrics:
        # Get all metrics from first entry
        if training_history:
            metrics = list(training_history[0]['metrics'].keys())
        else:
            print("No training history to plot")
            return
    
    epochs = [entry['epoch'] for entry in training_history]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        values = [entry['metrics'].get(metric, None) for entry in training_history]
        values = [v for v in values if v is not None]  # Filter None values
        
        if values:
            axes[i].plot(epochs[:len(values)], values, 'b-', label=metric)
            axes[i].set_title(metric)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric)
            axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def compare_checkpoints(checkpoint_paths: List[str], 
                       metric_names: List[str] = None) -> Dict[str, Any]:
    """Compare multiple checkpoints based on metrics"""
    comparison = {}
    
    for path in checkpoint_paths:
        checkpoint_name = Path(path).stem
        state = load_checkpoint_state(path)
        
        comparison[checkpoint_name] = {
            'epoch': state.get('epoch', 0),
            'metrics': state.get('metrics', {}),
            'best_metrics': state.get('best_metrics', {})
        }
    
    return comparison