"""
Training Script for Fixed Attention Baseline
Trains baseline models and evaluates efficiency metrics.
Implements the Phase 1 training pipeline from the project plan.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import argparse
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

# Import our custom modules
try:
    from fovea_lib.transforms import FovealTransform
    from fovea_lib.dataset_builder import create_baseline_dataloaders
    from fovea_lib.metrics import EfficiencyMetrics, ModelComparator, measure_model_efficiency
    from fovea_lib.baseline_model import create_baseline_model, FullResolutionBaseline
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the foveate directory and all files are in fovea_lib/")
    print("Run 'python experiments/setup_phase1.py' first to verify the setup.")
    exit(1)


class BaselineTrainer:
    """
    Trainer class for baseline foveal vision models.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 config: dict,
                 device: torch.device):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            test_loader: Test data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config['lr_step_size'], 
            gamma=config['lr_gamma']
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.best_accuracy = 0.0
        
        # Setup logging
        self.log_dir = config['log_dir']
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
    
    def train_epoch(self, epoch: int) -> tuple:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, (glimpses, full_images, labels) in enumerate(pbar):
            glimpses = glimpses.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(glimpses)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
            # Log to tensorboard
            if batch_idx % 100 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), step)
                self.writer.add_scalar('Train/BatchAcc', 100.*correct/total, step)
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, epoch: int) -> float:
        """
        Evaluate model on test set.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Test accuracy
        """
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for glimpses, full_images, labels in tqdm(self.test_loader, desc='Testing'):
                glimpses = glimpses.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(glimpses)
                loss = self.criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(self.test_loader)
        
        # Log to tensorboard
        self.writer.add_scalar('Test/Loss', avg_loss, epoch)
        self.writer.add_scalar('Test/Accuracy', accuracy, epoch)
        
        return accuracy
    
    def train(self, start_epoch: int = 0) -> dict:
        """
        Full training loop with checkpoint support.
        
        Args:
            start_epoch: Epoch to start training from (for resume)
            
        Returns:
            Training history dictionary
        """
        print(f"Training {self.model.__class__.__name__} for {self.config['epochs']} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if start_epoch > 0:
            print(f"🔄 Resuming training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, self.config['epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Evaluate
            test_acc = self.evaluate(epoch)
            self.test_accuracies.append(test_acc)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save best model
            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
                self.save_checkpoint(epoch, is_best=True)
            
            # Save periodic checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, is_periodic=True)
            
            # Always save latest checkpoint
            self.save_checkpoint(epoch, is_best=False, is_periodic=False)
            
            # Log epoch results
            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Train/EpochAcc', train_acc, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
            
            # Early stopping if accuracy plateaus (optional)
            if epoch > 10 and len(self.test_accuracies) >= 5:
                recent_accs = self.test_accuracies[-5:]
                if max(recent_accs) - min(recent_accs) < 0.5:  # Less than 0.5% improvement
                    print(f"🔄 Early stopping: accuracy plateaued at {test_acc:.2f}%")
                    break
        
        # Save final model
        self.save_checkpoint(self.config['epochs']-1, is_best=False)
        
        # Close tensorboard writer
        self.writer.close()
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'best_accuracy': self.best_accuracy
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, is_periodic: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'config': self.config,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies, 
            'test_accuracies': self.test_accuracies
        }
        
        if is_best:
            filepath = os.path.join(self.log_dir, 'best_model.pth')
            torch.save(checkpoint, filepath)
            print(f"💾 Best model saved with accuracy: {self.best_accuracy:.2f}%")
        
        if is_periodic:
            filepath = os.path.join(self.log_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, filepath)
            print(f"💾 Checkpoint saved: epoch {epoch}")
        
        # Always save latest checkpoint
        latest_filepath = os.path.join(self.log_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_filepath)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint and resume training."""
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training history
        self.best_accuracy = checkpoint['best_accuracy']
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.test_accuracies = checkpoint.get('test_accuracies', [])
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Checkpoint loaded! Resuming from epoch {start_epoch}")
        print(f"📊 Best accuracy so far: {self.best_accuracy:.2f}%")
        
        return start_epoch


def evaluate_efficiency_comparison(models_dict: dict, 
                                 test_loader: torch.utils.data.DataLoader,
                                 device: torch.device) -> dict:
    """
    Compare efficiency metrics across different models.
    
    Args:
        models_dict: Dictionary of {name: model} pairs
        test_loader: Test data loader
        device: Device to run on
        
    Returns:
        Comparison results dictionary
    """
    comparator = ModelComparator()
    
    print("\nEvaluating efficiency metrics...")
    
    for model_name, model in models_dict.items():
        print(f"Evaluating {model_name}...")
        
        # Measure efficiency
        metrics = measure_model_efficiency(model, test_loader, device, max_batches=50)
        
        # Add to comparator
        comparator.add_model(model_name)
        comparator.update_model(
            model_name,
            accuracy=metrics['overall_accuracy'],
            n_glimpses=metrics['mean_n_glimpses'],
            processing_time=metrics['mean_processing_time'],
            total_pixels=metrics['bandwidth_efficiency'] * metrics['mean_n_glimpses']
        )
    
    return comparator.compare_models()


def plot_training_curves(history: dict, save_path: str):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(history['train_losses'])
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(history['train_accuracies'], label='Train')
    ax2.plot(history['test_accuracies'], label='Test')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Baseline Foveal Vision Model')
    parser.add_argument('--n_glimpses', type=int, default=3, help='Number of glimpses')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--compare_full', action='store_true', help='Compare with full resolution')
    
    # Checkpoint support
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--auto_resume', action='store_true', help='Auto resume from latest checkpoint')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"baseline_g{args.n_glimpses}_{timestamp}")
    
    # Handle resume from checkpoint
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = args.resume
    elif args.auto_resume:
        # Look for latest checkpoint in most recent log directory
        pattern = f"baseline_g{args.n_glimpses}_*"
        matching_dirs = sorted([d for d in os.listdir(args.log_dir) 
                               if d.startswith(f"baseline_g{args.n_glimpses}_")])
        if matching_dirs:
            latest_dir = os.path.join(args.log_dir, matching_dirs[-1])
            latest_checkpoint = os.path.join(latest_dir, 'latest_checkpoint.pth')
            if os.path.exists(latest_checkpoint):
                resume_checkpoint = latest_checkpoint
                log_dir = latest_dir  # Use existing log directory
                print(f"🔄 Auto-resuming from: {resume_checkpoint}")
    
    # Ensure log directory exists and handle relative paths
    if not os.path.isabs(args.log_dir):
        # If relative path, make it relative to project root, not experiments/
        args.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', args.log_dir)
        log_dir = os.path.join(args.log_dir, f"baseline_g{args.n_glimpses}_{timestamp}")
    
    # Configuration
    config = {
        'n_glimpses': args.n_glimpses,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'lr_step_size': 15,
        'lr_gamma': 0.1,
        'log_dir': log_dir,
        'data_dir': args.data_dir
    }
    
    # Handle data directory path
    if not os.path.isabs(config['data_dir']):
        config['data_dir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', config['data_dir'])
    
    # Create datasets
    print("Loading datasets...")
    train_loader, test_loader = create_baseline_dataloaders(
        dataset_name='cifar10',
        data_dir=config['data_dir'],
        n_glimpses=config['n_glimpses'],
        batch_size=config['batch_size'],
        num_workers=4
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create and train baseline model
    print("\nCreating baseline model...")
    baseline_model = create_baseline_model(
        model_type='fixed',
        num_classes=10,
        n_glimpses=config['n_glimpses'],
        pretrained_backbone=False
    )
    
    # Train baseline
    trainer = BaselineTrainer(baseline_model, train_loader, test_loader, config, device)
    
    # Handle checkpoint resume
    start_epoch = 0
    if resume_checkpoint:
        start_epoch = trainer.load_checkpoint(resume_checkpoint)
        if start_epoch is False:
            print("❌ Failed to load checkpoint, starting from scratch")
            start_epoch = 0
    
    # Train the model
    history = trainer.train(start_epoch)
    
    # Plot training curves
    plot_path = os.path.join(log_dir, 'training_curves.png')
    plot_training_curves(history, plot_path)
    
    # Save results
    results = {
        'config': config,
        'history': history,
        'best_accuracy': history['best_accuracy']
    }
    
    results_path = os.path.join(log_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best accuracy: {history['best_accuracy']:.2f}%")
    print(f"Results saved to: {log_dir}")
    
    # Efficiency comparison
    models_to_compare = {'baseline': baseline_model}
    
    if args.compare_full:
        print("\nTraining full resolution baseline for comparison...")
        full_res_model = FullResolutionBaseline(num_classes=10)
        
        # Quick training for comparison (fewer epochs)
        config_full = config.copy()
        config_full['epochs'] = 10
        config_full['log_dir'] = os.path.join(log_dir, 'full_res')
        
        trainer_full = BaselineTrainer(full_res_model, train_loader, test_loader, config_full, device)
        trainer_full.train()
        
        models_to_compare['full_resolution'] = full_res_model
    
    # Evaluate efficiency
    efficiency_results = evaluate_efficiency_comparison(models_to_compare, test_loader, device)
    
    print("\nEfficiency Comparison:")
    for model_name, metrics in efficiency_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {metrics['mean_accuracy']:.2f}%")
        print(f"  Information Efficiency: {metrics['information_efficiency']:.4f}")
        print(f"  Bandwidth Efficiency: {metrics['bandwidth_efficiency']:.0f} pixels/glimpse")
        print(f"  Processing Time: {metrics['mean_processing_time']:.4f}s")
    
    # Save efficiency results
    efficiency_path = os.path.join(log_dir, 'efficiency_comparison.json')
    with open(efficiency_path, 'w') as f:
        json.dump(efficiency_results, f, indent=2)


if __name__ == "__main__":
    main()