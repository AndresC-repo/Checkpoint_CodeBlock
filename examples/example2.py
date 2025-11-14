import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from tqdm import tqdm

# Add the parent directory to Python path to import Checkpoint_CodeBlock
sys.path.append('/app')

from wandb_integration import setup_wandb_training
from examples.utils import SimpleCNN, get_data_loaders, train_epoch, validate_model

# Set random seeds for reproducibility
torch.manual_seed(42)

def main():
    print("🚀 Starting Checkpoint_CodeBlock with W&B Integration Demo!")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model and data
    model = SimpleCNN().to(device)
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize checkpointing system WITH W&B
    print("📊 Initializing checkpoint system with W&B integration...")
    tm, wandb_int = setup_wandb_training(
        model=model,
        optimizer=optimizer,
        wandb_project="checkpoint_codeblock_demo",  # Your W&B project name
        wandb_entity=None,  # Your W&B team name (optional)
        checkpoint_dir="mnist_wandb_experiment",
        model_name="simple_cnn_wandb",
        metrics_to_track={
            'train_loss': 'min', 
            'val_loss': 'min', 
            'val_accuracy': 'max',
            'train_accuracy': 'max'
        },
        hyperparams={
            'lr': 0.001,
            'batch_size': 64,
            'model': 'SimpleCNN',
            'dataset': 'MNIST',
            'optimizer': 'Adam',
            'criterion': 'CrossEntropyLoss'
        }
    )
    
    print(f"🔄 Starting from epoch: {tm.current_epoch}")
    status = tm.get_training_status()
    print(f"📈 Training status: Can resume = {status['can_resume']}")
    
    # Training loop
    total_epochs = 10  # Small number for demo
    
    print(f"\n🎯 Training for {total_epochs} epochs with W&B logging...")
    for epoch in tqdm(range(tm.current_epoch, total_epochs)):
        print(f"\n--- Epoch {epoch + 1}/{total_epochs} ---")
        
        # Train
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        
        print(f"📈 Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"📊 Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # Save checkpoint AND log to W&B (all in one!)
        metrics = {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss, 
            'val_accuracy': val_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr']  # Log learning rate too
        }
        
        saved_paths = wandb_int.save_and_log(metrics, epoch=epoch+1)
        print(f"💾 Checkpoints saved: {list(saved_paths.keys())}")
        print(f"📡 Metrics logged to W&B!")
    
    # Demonstrate checkpoint features
    print("\n" + "="*60)
    print("✅ W&B INTEGRATION DEMONSTRATION COMPLETE!")
    print("="*60)
    
    # Show training status
    final_status = tm.get_training_status()
    print(f"📈 Final Training Status:")
    print(f"   - Current epoch: {final_status['current_epoch']}")
    print(f"   - Best metrics: {final_status['best_metrics']}")
    print(f"   - Total checkpoints in history: {final_status['total_checkpoints']}")
    
    # Show what files were created
    print(f"\n📁 Checkpoint files created:")
    checkpoint_dir = "mnist_wandb_experiment"
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            if file.endswith(".pt"):
                file_size = os.path.getsize(f"{checkpoint_dir}/{file}") / 1024  # KB
                print(f"   - {file} ({file_size:.1f} KB)")
    else:
        print("   No checkpoint directory found!")
    
    # Demonstrate loading best model
    print(f"\n🔄 Loading best model based on 'val_accuracy':")
    try:
        best_state = tm.load_best_model('val_accuracy')
        best_accuracy = tm.get_best_metric_value('val_accuracy')
        print(f"   ✅ Best validation accuracy: {best_accuracy:.2f}%")
        
        # Test the loaded best model
        print(f"\n🧪 Testing best model on test set:")
        test_loss, test_accuracy = validate_model(model, test_loader, criterion, device)
        print(f"   📊 Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        # Log final test results to W&B
        wandb_int.log_metrics({
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        })
    except Exception as e:
        print(f"   ❌ Error loading best model: {e}")
    
    # Demonstrate resume capability with W&B
    print(f"\n" + "="*60)
    print("🔄 RESUME TRAINING WITH W&B DEMONSTRATION")
    print("="*60)
    
    # Create a new training manager with same settings - it should resume
    print("Creating new training manager (simulating resume after restart)...")
    model_resume = SimpleCNN().to(device)
    optimizer_resume = optim.Adam(model_resume.parameters(), lr=0.001)
    
    # Resume with W&B integration
    tm_resume, wandb_int_resume = setup_wandb_training(
        model=model_resume,
        optimizer=optimizer_resume,
        wandb_project="checkpoint_codeblock_demo_resume",  # Different project for resume demo
        checkpoint_dir="mnist_wandb_experiment",  # Same directory
        model_name="simple_cnn_wandb"             # Same model name
    )
    
    status_resume = tm_resume.get_training_status()
    print(f"Resumed training status:")
    print(f"  - Starting from epoch: {tm_resume.current_epoch}")
    print(f"  - Best metrics preserved: {status_resume['best_metrics']}")
    print(f"  - Can resume: {status_resume['can_resume']}")
    
    # Train one more epoch to show resume works with W&B
    if tm_resume.current_epoch < total_epochs:
        epoch = tm_resume.current_epoch
        print(f"\nTraining one more epoch to demonstrate resume with W&B...")
        train_loss, train_accuracy = train_epoch(model_resume, train_loader, criterion, optimizer_resume, device)
        val_loss, val_accuracy = validate_model(model_resume, val_loader, criterion, device)
        
        metrics = {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'learning_rate': optimizer_resume.param_groups[0]['lr']
        }
        
        saved_paths = wandb_int_resume.save_and_log(metrics, epoch=epoch+1)
        print(f"Resumed training - Epoch {epoch + 1} completed!")
        print(f"New best validation accuracy: {tm_resume.get_best_metric_value('val_accuracy'):.2f}%")
    
    # Finalize W&B runs
    print(f"\n🎬 Finalizing W&B runs...")
    wandb_int.finish()
    if 'wandb_int_resume' in locals():
        wandb_int_resume.finish()
    
    print("\n" + "="*60)
    print("🎉 Checkpoint_CodeBlock W&B Integration Demo Completed!")
    print("="*60)

if __name__ == "__main__":
    main()