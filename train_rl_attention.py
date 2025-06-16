"""
Train RL attention model for foveal pattern emergence.
"""

import torch
import torch.optim as optim
import argparse
import os
from tqdm import tqdm

from fovea_lib.rl_attention import RecurrentAttentionModel, REINFORCETrainer
from fovea_lib.dataset_builder import create_cifar10_loaders, create_mnist_loaders
from fovea_lib.metrics import TrainingMonitor


def train_epoch(trainer, dataloader, epoch):
    """Train for one epoch."""
    total_stats = {
        'total_loss': 0, 'classification_loss': 0,
        'reinforce_loss': 0, 'accuracy': 0, 'avg_reward': 0
    }
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(trainer.device)
        labels = labels.to(trainer.device)
        
        stats = trainer.train_step(images, labels)
        
        for key in total_stats:
            total_stats[key] += stats[key]
        
        if batch_idx % 100 == 0:
            pbar.set_postfix({
                'Loss': f"{stats['total_loss']:.3f}",
                'Acc': f"{stats['accuracy']:.3f}",
                'Reward': f"{stats['avg_reward']:.3f}"
            })
    
    # Average over batches
    for key in total_stats:
        total_stats[key] /= len(dataloader)
    
    return total_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'mnist'], default='cifar10')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_glimpses', type=int, default=6)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--save_dir', default='./results')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Training on {device}")
    
    # Create datasets
    if args.dataset == 'cifar10':
        train_loader, test_loader = create_cifar10_loaders(
            batch_size=args.batch_size
        )
        num_classes = 10
    else:
        train_loader, test_loader = create_mnist_loaders(
            batch_size=args.batch_size
        )
        num_classes = 10
    
    # Create model
    model = RecurrentAttentionModel(
        glimpse_size=64,
        feature_dim=256,
        hidden_dim=256,
        num_classes=num_classes,
        max_glimpses=args.max_glimpses
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Trainer
    trainer = REINFORCETrainer(model, optimizer, device)
    
    # Monitor
    monitor = TrainingMonitor()
    
    # Training loop
    best_accuracy = 0
    os.makedirs(args.save_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        # Train
        train_stats = train_epoch(trainer, train_loader, epoch)
        
        # Evaluate
        eval_results = trainer.evaluate(test_loader)
        
        # Monitor emergence
        monitor.update(epoch, eval_results, eval_results['attention_patterns'][:100])
        
        # Print progress
        print(f"Epoch {epoch}: "
              f"Train Loss: {train_stats['total_loss']:.3f}, "
              f"Test Acc: {eval_results['accuracy']:.3f}")
        
        # Save best model
        if eval_results['accuracy'] > best_accuracy:
            best_accuracy = eval_results['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'args': args
            }, os.path.join(args.save_dir, 'best_model.pth'))
        
        # Plot emergence curves every 10 epochs
        if (epoch + 1) % 10 == 0:
            fig = monitor.plot_emergence_curves()
            if fig:
                fig.savefig(os.path.join(args.save_dir, f'emergence_epoch_{epoch}.png'))
                fig.close()
    
    # Final analysis
    print("\nTraining Summary:")
    summary = monitor.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    
    # Save final results
    torch.save({
        'monitor': monitor,
        'summary': summary,
        'args': args
    }, os.path.join(args.save_dir, 'training_results.pth'))
    
    print(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()