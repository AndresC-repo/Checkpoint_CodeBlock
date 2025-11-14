"""
Checkpoint_CodeBlock - A simple but powerful checkpointing system for PyTorch
"""

from .core import BaseCheckpoint, SmartCheckpoint
from .managers import TrainingManager, CheckpointConfig
from .wandb_integration import WandBIntegration

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = [
    'BaseCheckpoint', 
    'SmartCheckpoint', 
    'TrainingManager', 
    'CheckpointConfig',
    'WandBIntegration'
]