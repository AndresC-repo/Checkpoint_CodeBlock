import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import logging
from core import SmartCheckpoint, CheckpointConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingManager:
    """
    Main training manager that handles checkpointing, resuming, and metric tracking
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 config: CheckpointConfig,
                 schedulers: Dict[str, Any] = None,
                 hyperparams: Dict[str, Any] = None):
        
        self.model = model
        self.optimizer = optimizer
        self.schedulers = schedulers or {}
        self.hyperparams = hyperparams or {}
        
        # Initialize checkpoint system
        self.checkpoint = SmartCheckpoint(config)
        self.current_epoch = 0
        self.device = next(model.parameters()).device
        
        # Resume if possible
        self._resume_training()
    
    def _resume_training(self):
        """Resume training from checkpoint if available"""
        if self.checkpoint.can_resume():
            logger.info("Resuming training from checkpoint...")
            state_dict = self.checkpoint.load_checkpoint(device=self.device)
            
            # Load model state
            self.model.load_state_dict(state_dict['model_state'])
            
            # Load optimizer state
            if 'optimizer_state' in state_dict and hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(state_dict['optimizer_state'])
            
            # Load scheduler states
            if 'schedulers' in state_dict and self.schedulers:
                for name, scheduler_data in state_dict['schedulers'].items():
                    if name in self.schedulers:
                        self.schedulers[name].load_state_dict(scheduler_data['state'])
            
            # Set current epoch
            self.current_epoch = state_dict.get('epoch', 0) + 1
            
            logger.info(f"Resumed from epoch {self.current_epoch}")
            logger.info(f"Best metrics: {state_dict.get('best_metrics', {})}")
        else:
            logger.info("Starting fresh training")
            self.current_epoch = 0
    
    def save(self, 
             metrics: Dict[str, float], 
             epoch: Optional[int] = None,
             force_save: bool = False) -> Dict[str, str]:
        """
        Save checkpoint with current state
        
        Args:
            metrics: Dictionary of current metrics
            epoch: Current epoch (uses internal counter if None)
            force_save: Force save even if metrics didn't improve
            
        Returns:
            Dictionary of saved checkpoint paths
        """
        if epoch is None:
            epoch = self.current_epoch
            self.current_epoch += 1
        
        saved_paths = self.checkpoint.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            schedulers=self.schedulers,
            epoch=epoch,
            metrics=metrics,
            hyperparams=self.hyperparams,
            force_save=force_save
        )
        
        # Log improvements
        improvements = self.checkpoint.should_save_best(metrics)
        for metric, improved in improvements.items():
            if improved:
                best_value = self.checkpoint.best_metrics[metric]
                logger.info(f"🎉 New best {metric}: {best_value:.6f}")
        
        logger.info(f"Checkpoint saved (epoch {epoch})")
        return saved_paths
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'current_epoch': self.current_epoch,
            'best_metrics': self.checkpoint.best_metrics,
            'total_checkpoints': len(self.checkpoint.training_history),
            'can_resume': self.checkpoint.can_resume(),
            'config': self.checkpoint.config
        }
    
    def get_best_metric_value(self, metric_name: str) -> Optional[float]:
        """Get the best value for a specific metric"""
        return self.checkpoint.best_metrics.get(metric_name)
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get complete training history"""
        return self.checkpoint.training_history
    
    def load_best_model(self, metric_name: str = None) -> Dict[str, Any]:
        """
        Load the best model based on a specific metric
        
        Args:
            metric_name: Metric to use for best model selection. 
                        If None, uses the first metric in config
        
        Returns:
            State dictionary of the best model
        """
        if metric_name is None:
            metric_name = list(self.checkpoint.config.metrics_to_track.keys())[0]
        
        best_checkpoint_path = (
            self.checkpoint.filepath / 
            f"{self.checkpoint.config.model_name}_best_{metric_name}.pt"
        )
        
        if not best_checkpoint_path.exists():
            available = list(self.checkpoint.best_metrics.keys())
            raise ValueError(
                f"No best model found for metric '{metric_name}'. "
                f"Available metrics: {available}"
            )
        
        state_dict = torch.load(best_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict['model_state'])
        
        logger.info(f"Loaded best model for {metric_name}: {self.checkpoint.best_metrics[metric_name]:.6f}")
        return state_dict

# Convenience function for quick setup
def create_training_manager(model: nn.Module,
                          optimizer: torch.optim.Optimizer,
                          checkpoint_dir: str = "checkpoints",
                          model_name: str = "model",
                          metrics_to_track: Dict[str, str] = None,
                          hyperparams: Dict[str, Any] = None) -> TrainingManager:
    """
    Quick convenience function to create a TrainingManager
    """
    config = CheckpointConfig(
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        metrics_to_track=metrics_to_track or {'loss': 'min'}
    )
    
    return TrainingManager(
        model=model,
        optimizer=optimizer,
        config=config,
        hyperparams=hyperparams
    )