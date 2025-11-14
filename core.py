import torch
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import copy

@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management"""
    checkpoint_dir: str = "checkpoints"
    model_name: str = "model"
    metrics_to_track: Dict[str, str] = None  # {'accuracy': 'max', 'loss': 'min'}
    save_optimizer: bool = True
    save_schedulers: bool = True
    save_hyperparams: bool = True
    
    def __post_init__(self):
        if self.metrics_to_track is None:
            self.metrics_to_track = {'loss': 'min'}

class BaseCheckpoint:
    """Simple checkpoint base class"""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.filepath = Path(config.checkpoint_dir)
        self.filepath.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.filepath / f"{config.model_name}.pt"
        
    def save(self, state_dict: Dict) -> str:
        """Save checkpoint to file"""
        torch.save(state_dict, self.checkpoint_file)
        return str(self.checkpoint_file)
        
    def load(self, device: str = 'cpu') -> Dict:
        """Load checkpoint from file"""
        return torch.load(self.checkpoint_file, map_location=device)
    
    def exists(self) -> bool:
        """Check if checkpoint exists"""
        return self.checkpoint_file.exists()
    
    def get_latest_path(self) -> Path:
        """Get path to latest checkpoint"""
        return self.filepath / f"{self.config.model_name}_latest.pt"

class SmartCheckpoint(BaseCheckpoint):
    """Enhanced checkpoint with metric tracking and resume capability"""
    
    def __init__(self, config: CheckpointConfig):
        super().__init__(config)
        self.best_metrics = {}
        self.training_history = []
        
    def create_state_dict(self, 
                         model, 
                         optimizer=None, 
                         schedulers=None, 
                         epoch=0, 
                         metrics=None,
                         hyperparams=None) -> Dict[str, Any]:
        """Create comprehensive state dictionary"""
        
        state_dict = {
            'epoch': epoch,
            'model_state': copy.deepcopy(model.state_dict()),
            'metrics': metrics or {},
            'best_metrics': copy.deepcopy(self.best_metrics),
            'history': copy.deepcopy(self.training_history),
            'config': asdict(self.config),
            'model_class': model.__class__.__name__
        }
        
        # Add optimizer state if requested
        if self.config.save_optimizer and optimizer is not None:
            state_dict['optimizer_state'] = copy.deepcopy(optimizer.state_dict())
            state_dict['optimizer_class'] = optimizer.__class__.__name__
        
        # Add scheduler states if requested
        if self.config.save_schedulers and schedulers is not None:
            state_dict['schedulers'] = {}
            for name, scheduler in schedulers.items():
                state_dict['schedulers'][name] = {
                    'state': copy.deepcopy(scheduler.state_dict()),
                    'class': scheduler.__class__.__name__
                }
        
        # Add hyperparameters
        if self.config.save_hyperparams and hyperparams is not None:
            state_dict['hyperparams'] = copy.deepcopy(hyperparams)
        
        return state_dict
    
    def should_save_best(self, current_metrics: Dict[str, float]) -> Dict[str, bool]:
        """Determine which metrics improved and should trigger a save"""
        improvements = {}
        
        for metric, value in current_metrics.items():
            if metric not in self.config.metrics_to_track:
                continue
                
            if metric not in self.best_metrics:
                # First time seeing this metric
                self.best_metrics[metric] = value
                improvements[metric] = True
            else:
                direction = self.config.metrics_to_track[metric]
                current_best = self.best_metrics[metric]
                
                if direction == 'max' and value > current_best:
                    improvements[metric] = True
                    self.best_metrics[metric] = value
                elif direction == 'min' and value < current_best:
                    improvements[metric] = True
                    self.best_metrics[metric] = value
                else:
                    improvements[metric] = False
                    
        return improvements
    
    def save_checkpoint(self, 
                       model, 
                       optimizer=None, 
                       schedulers=None, 
                       epoch=0, 
                       metrics=None,
                       hyperparams=None,
                       force_save=False) -> Dict[str, str]:
        """
        Save checkpoint with automatic best-model tracking
        
        Returns:
            Dict with paths to saved checkpoints
        """
        metrics = metrics or {}
        saved_paths = {}
        
        # Create state dict
        state_dict = self.create_state_dict(
            model, optimizer, schedulers, epoch, metrics, hyperparams
        )
        
        # Update history
        history_entry = {'epoch': epoch, 'metrics': copy.deepcopy(metrics)}
        self.training_history.append(history_entry)
        
        # Always save latest
        latest_path = self.get_latest_path()
        torch.save(state_dict, latest_path)
        saved_paths['latest'] = str(latest_path)
        
        # Save best models for improved metrics
        if metrics:
            improvements = self.should_save_best(metrics)
            for metric, improved in improvements.items():
                if improved or force_save:
                    best_path = self.filepath / f"{self.config.model_name}_best_{metric}.pt"
                    torch.save(state_dict, best_path)
                    saved_paths[f'best_{metric}'] = str(best_path)
        
        # Save main checkpoint
        main_save_path = self.save(state_dict)
        saved_paths['main'] = main_save_path
        
        return saved_paths
    
    def load_checkpoint(self, device='cpu') -> Dict[str, Any]:
        """Load checkpoint and restore internal state"""
        state_dict = self.load(device)
        
        # Restore internal state
        self.best_metrics = state_dict.get('best_metrics', {})
        self.training_history = state_dict.get('history', [])
        
        return state_dict
    
    def can_resume(self) -> bool:
        """Check if we can resume training from existing checkpoint"""
        return self.exists()
    
    def get_resume_info(self) -> Dict[str, Any]:
        """Get information about resume state"""
        if not self.exists():
            return {'can_resume': False}
        
        state_dict = self.load()
        return {
            'can_resume': True,
            'last_epoch': state_dict.get('epoch', 0),
            'best_metrics': state_dict.get('best_metrics', {}),
            'total_epochs': len(state_dict.get('history', [])),
            'checkpoint_path': str(self.checkpoint_file)
        }