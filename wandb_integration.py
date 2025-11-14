import wandb
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
from core import SmartCheckpoint, CheckpointConfig
from managers import TrainingManager

class WandBIntegration:
    """
    Integration with Weights & Biases for experiment tracking
    """
    
    def __init__(self, 
                 training_manager: TrainingManager,
                 wandb_project: str,
                 wandb_entity: Optional[str] = None,
                 log_checkpoints: bool = True,
                 log_artifacts: bool = True):
        
        self.tm = training_manager
        self.log_checkpoints = log_checkpoints
        self.log_artifacts = log_artifacts
        
        # Initialize W&B
        self._init_wandb(wandb_project, wandb_entity)
    
    def _init_wandb(self, project: str, entity: Optional[str] = None):
        """Initialize Weights & Biases"""
        wandb.init(project=project, entity=entity)
        
        # Log hyperparameters
        if self.tm.hyperparams:
            wandb.config.update(self.tm.hyperparams)
        
        # Log model architecture
        wandb.watch(self.tm.model, log='all', log_freq=100)
    
    def log_metrics(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        """Log metrics to W&B"""
        if epoch is not None:
            metrics['epoch'] = epoch
        
        wandb.log(metrics)
        
        # Log best metrics
        best_metrics = {f'best_{k}': v for k, v in self.tm.checkpoint.best_metrics.items()}
        wandb.log(best_metrics)
    
    def save_and_log(self, 
                    metrics: Dict[str, float], 
                    epoch: Optional[int] = None,
                    force_save: bool = False) -> Dict[str, str]:
        """
        Save checkpoint and log to W&B
        
        Returns:
            Dictionary of saved checkpoint paths
        """
        # Save checkpoint
        saved_paths = self.tm.save(metrics, epoch, force_save)
        
        # Log to W&B
        self.log_metrics(metrics, epoch)
        
        # Log checkpoints as artifacts if requested
        if self.log_artifacts:
            for checkpoint_type, path in saved_paths.items():
                if 'best' in checkpoint_type or force_save:
                    artifact = wandb.Artifact(
                        name=f"{self.tm.checkpoint.config.model_name}_{checkpoint_type}",
                        type="model",
                        description=f"{checkpoint_type} model checkpoint"
                    )
                    artifact.add_file(path)
                    wandb.log_artifact(artifact)
        
        return saved_paths
    
    def log_training_status(self):
        """Log current training status to W&B"""
        status = self.tm.get_training_status()
        
        # Log as summary statistics
        for metric, value in status['best_metrics'].items():
            wandb.run.summary[f'best_{metric}'] = value
        
        wandb.run.summary['total_epochs'] = status['current_epoch']
    
    def finish(self):
        """Finish W&B run and log final status"""
        self.log_training_status()
        wandb.finish()

# Convenience function for quick W&B setup
def setup_wandb_training(model: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        wandb_project: str,
                        wandb_entity: Optional[str] = None,
                        checkpoint_dir: str = "checkpoints",
                        model_name: str = "model",
                        metrics_to_track: Dict[str, str] = None,
                        hyperparams: Dict[str, Any] = None) -> tuple:
    """
    Quick setup for training with W&B integration
    
    Returns:
        Tuple of (TrainingManager, WandBIntegration)
    """
    from managers import create_training_manager
    
    # Create training manager
    tm = create_training_manager(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        metrics_to_track=metrics_to_track,
        hyperparams=hyperparams
    )
    
    # Create W&B integration
    wandb_int = WandBIntegration(
        training_manager=tm,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity
    )
    
    return tm, wandb_int