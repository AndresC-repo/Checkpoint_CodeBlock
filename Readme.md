# Checkpoint_CodeBlock 🚀

A simple but powerful checkpointing system for PyTorch that handles model saving, training resumption, metric tracking, and W&B integration with minimal code.

### Quick Start

## Instalation
```
pip install torch wandb matplotlib
git clone <your-repo>
cd Checkpoint_CodeBlock
```

## Basic Usage

### 1. Minimal Integration (3 lines of code)
```
import torch
import torch.nn as nn
from Checkpoint_CodeBlock import create_training_manager

# Your existing model and optimizer
model = nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters())

# Add checkpointing with one line
tm = create_training_manager(
    model=model,
    optimizer=optimizer,
    model_name="my_model",
    metrics_to_track={'loss': 'min', 'accuracy': 'max'}
)

# Your training loop (minimal changes)
for epoch in range(100):
    train_loss = ...  # Your training code
    val_accuracy = ...  # Your validation code
    
    # Save checkpoint (automatically tracks best models)
    tm.save({
        'loss': train_loss,
        'accuracy': val_accuracy
    })
```

### 2. With W&B Integration
```
from Checkpoint_CodeBlock import setup_wandb_training

# Setup training with W&B in one line
tm, wandb_int = setup_wandb_training(
    model=model,
    optimizer=optimizer,
    wandb_project="my_project",
    model_name="experiment_1",
    hyperparams={'lr': 0.001, 'batch_size': 32}
)

for epoch in range(100):
    metrics = {
        'loss': train_loss,
        'accuracy': val_accuracy,
        'learning_rate': current_lr
    }
    
    # Save and automatically log to W&B
    wandb_int.save_and_log(metrics)
```

## Features ✨

### ✅ Automatic Best Model Tracking
```
# Saves separate models for each metric:
# - my_model_best_loss.pt
# - my_model_best_accuracy.pt  
# - my_model_latest.pt

tm.save({'loss': 0.1, 'accuracy': 0.95})
# 🎉 New best loss: 0.100000
# 🎉 New best accuracy: 0.950000
```

### ✅ Seamless Training Resumption
```
# If checkpoints exist, automatically resumes
tm = create_training_manager(model, optimizer, model_name="my_model")
print(f"Resumed from epoch {tm.current_epoch}")
print(f"Best accuracy: {tm.get_best_metric_value('accuracy')}")

# Continue training exactly where you left off
for epoch in range(tm.current_epoch, 100):
    # ... training code
    tm.save(metrics)
```

### ✅ W&B Integration
```
# Automatic experiment tracking
tm, wandb_int = setup_wandb_training(...)

# Logs to W&B:
# - Metrics every epoch
# - Best metrics as summary
# - Model checkpoints as artifacts
# - Hyperparameters
```

### ✅ Flexible Metric Tracking

```
# Track any metrics with custom directions
metrics_to_track = {
    'loss': 'min',           # Lower is better
    'accuracy': 'max',       # Higher is better  
    'precision': 'max',
    'iou': 'max',
    'f1_score': 'max'
}
```

### Core Classes
##### TrainingManager
-   save(metrics, epoch=None): Save checkpoint with metrics

-   load_best_model(metric_name): Load best model for specific metric

-   get_training_status(): Get current training state

-   get_best_metric_value(metric_name): Get best value for metric

##### WandBIntegration
-   save_and_log(metrics, epoch=None): Save checkpoint and log to W&B

-   log_metrics(metrics, epoch): Log metrics to W&B

-   finish(): Complete W&B run

#### Utility Functions
Quick setup for basic training.

```
create_training_manager()
```

All-in-one setup with W&B integration.
```
setup_wandb_training()
```


