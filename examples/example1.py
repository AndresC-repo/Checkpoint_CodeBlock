import torch
import torch.nn as nn
import torch.optim as optim
import os
from managers import create_training_manager

from examples.utils import SimpleCNN, get_data_loaders, train_epoch, validate_model

# Set random seeds for reproducibility
torch.manual_seed(42)


def main():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model and data
    model = SimpleCNN().to(device)
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize checkpointing system
    tm = create_training_manager(
        model=model,
        optimizer=optimizer,
        checkpoint_dir="mnist_experiment",
        wandb_project="my_research",
        wandb_entity="my_team",  # Optional
        model_name="simple_cnn",
        metrics_to_track={'train_loss': 'min', 'val_loss': 'min', 'val_accuracy': 'max'},
        hyperparams={
            'lr': 0.001,
            'batch_size': 64,
            'model': 'SimpleCNN',
            'dataset': 'MNIST'
        }
    )
    
    print(f"Starting from epoch: {tm.current_epoch}")
    print(f"Training status: {tm.get_training_status()}")
    
    # Training loop
    total_epochs = 5  # Small number for demo
    
    for epoch in range(tm.current_epoch, total_epochs):
        print(f"\n--- Epoch {epoch + 1}/{total_epochs} ---")
        
        # Train
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # Save checkpoint (this is where the magic happens!)
        saved_paths = tm.save({
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss, 
            'val_accuracy': val_accuracy
        })
        
        print(f"Checkpoints saved: {list(saved_paths.keys())}")
    
    # Demonstrate checkpoint features
    print("\n" + "="*50)
    print("CHECKPOINT SYSTEM DEMONSTRATION")
    print("="*50)
    
    # Show training status
    status = tm.get_training_status()
    print(f"Final Status:")
    print(f"  - Current epoch: {status['current_epoch']}")
    print(f"  - Best metrics: {status['best_metrics']}")
    print(f"  - Total checkpoints in history: {status['total_checkpoints']}")
    
    # Show what files were created
    print(f"\nCheckpoint files created in 'mnist_experiment/':")
    for file in os.listdir("mnist_experiment"):
        if file.endswith(".pt"):
            file_size = os.path.getsize(f"mnist_experiment/{file}") / 1024  # KB
            print(f"  - {file} ({file_size:.1f} KB)")
    
    # Demonstrate loading best model
    print(f"\nLoading best model based on 'val_accuracy':")
    best_state = tm.load_best_model('val_accuracy')
    best_accuracy = tm.get_best_metric_value('val_accuracy')
    print(f"  Best validation accuracy: {best_accuracy:.2f}%")
    
    # Test the loaded best model
    print(f"\nTesting best model on test set:")
    test_loss, test_accuracy = validate_model(model, test_loader, criterion, device)
    print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    # Demonstrate resume capability
    print(f"\n" + "="*50)
    print("RESUME TRAINING DEMONSTRATION")
    print("="*50)
    
    # Create a new training manager with same settings - it should resume
    print("Creating new training manager (simulating resume after restart)...")
    model_resume = SimpleCNN().to(device)
    optimizer_resume = optim.Adam(model_resume.parameters(), lr=0.001)
    
    tm_resume = create_training_manager(
        model=model_resume,
        optimizer=optimizer_resume,
        checkpoint_dir="mnist_experiment",  # Same directory
        model_name="simple_cnn"             # Same model name
    )
    
    status_resume = tm_resume.get_training_status()
    print(f"Resumed training status:")
    print(f"  - Starting from epoch: {tm_resume.current_epoch}")
    print(f"  - Best metrics preserved: {status_resume['best_metrics']}")
    print(f"  - Can resume: {status_resume['can_resume']}")
    
    # Train one more epoch to show resume works
    if tm_resume.current_epoch < total_epochs:
        epoch = tm_resume.current_epoch
        print(f"\nTraining one more epoch to demonstrate resume...")
        train_loss, train_accuracy = train_epoch(model_resume, train_loader, criterion, optimizer_resume, device)
        val_loss, val_accuracy = validate_model(model_resume, val_loader, criterion, device)
        
        saved_paths = tm_resume.save({
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })
        print(f"Resumed training - Epoch {epoch + 1} completed!")
        print(f"New best validation accuracy: {tm_resume.get_best_metric_value('val_accuracy'):.2f}%")

if __name__ == "__main__":
    main()