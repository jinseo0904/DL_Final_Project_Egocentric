import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import argparse
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR
from collections import Counter

# Import models and dataset
from video_vit import VideoViT
from EgoVideoDataset import create_video_dataloaders_subset, create_video_dataloaders_selected_classes
from transformers import ViTModel, ViTConfig
from pytorchvideo.models.hub import i3d_r50
from train_ego_vit_pretrained import ImprovedVideoViT, PretrainedVideoViT
from timesformer_classifier import TimeSformerVideoClassifier

# Parse arguments
parser = argparse.ArgumentParser(description='Quick test of VideoViT on a subset of EgoVideo data')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--num_frames', type=int, default=8, help='Number of frames')
parser.add_argument('--frame_size', type=int, default=96, help='Frame size')
parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
parser.add_argument('--depth', type=int, default=4, help='Transformer depth')
parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training')
parser.add_argument('--top_n', type=int, default=10, help='Number of top classes to use')
parser.add_argument('--subset', type=float, default=0.1, help='Fraction of dataset to use')
parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
parser.add_argument('--model_type', type=str, default='vit', 
                    choices=['vit', 'i3d', 'videovit', 'timesformer'],
                    help='Type of model to use')
parser.add_argument('--freeze_backbone', action='store_true',
                    help='Freeze pretrained backbone')
parser.add_argument('--class_file', type=str, default='selected_classes_15.txt',
                    help='File containing selected classes')
# Add new argument
parser.add_argument('--no_augment', action='store_true', help='Disable data augmentation')
# Add these new arguments to your parser
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--use_class_weights', action='store_true', help='Use class-weighted loss')
parser.add_argument('--temporal_model', type=str, default='attn', 
                    choices=['attn', 'convgru'], help='Temporal model type')
parser.add_argument('--dynamic_unfreeze', action='store_true', 
                    help='Dynamically unfreeze backbone layers')
                    # Add to argument parser
parser.add_argument('--checkpoint_dir', type=str, default='/scratch/kim.jinseo/checkpoints', 
                    help='Directory to save checkpoints')
parser.add_argument('--resume', type=str, default='', 
                    help='Path to checkpoint to resume training from')
parser.add_argument('--save_freq', type=int, default=1, 
                    help='Save checkpoint every N epochs')
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.cuda.empty_cache()



def train_quick_test(args):
    # Create subset dataloaders
    # train_loader, val_loader, simplified_action_to_idx = create_video_dataloaders_subset(
    #     batch_size=args.batch_size,
    #     num_frames=args.num_frames,
    #     frame_size=(args.frame_size, args.frame_size),
    #     num_workers=2,
    #     top_n_classes=args.top_n,
    #     subset_fraction=args.subset
    # )
    
    # # Get number of classes
    # num_classes = len(simplified_action_to_idx)
    # print(f"Training quick test with {num_classes} classes")

    # In train_quick_test function
    train_loader, val_loader, action_to_idx = create_video_dataloaders_selected_classes(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        frame_size=(args.frame_size, args.frame_size),
        num_workers=getattr(args, 'num_workers', 2),
        class_file=args.class_file,
        subset_fraction=args.subset,
        use_augmentation=not args.no_augment  # Enable augmentation by default
    )

    # Get number of classes
    num_classes = len(action_to_idx)
    print(f"Training with {num_classes} selected diverse classes")
            
    # Create model with the chosen architecture
    if args.model_type == 'vit':
        if args.temporal_model == 'convgru':
            model = ImprovedVideoViT(
                num_classes=num_classes,
                num_frames=args.num_frames,
                freeze_backbone=args.freeze_backbone,
                dropout_rate=args.dropout
            )
        else:
            model = PretrainedVideoViT(
                num_classes=num_classes,
                num_frames=args.num_frames,
                freeze_backbone=args.freeze_backbone,
                dropout_rate=args.dropout
            )
    elif args.model_type == 'i3d':
        model = PretrainedI3D(
            num_classes=num_classes,
            freeze_backbone=args.freeze_backbone  # Use the argument value
        )
    elif args.model_type == 'timesformer':
        model = TimeSformerVideoClassifier(
            num_classes=num_classes,
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone
    )
    else:
        # Your existing VideoViT
        model = VideoViT(
            video_size=(args.num_frames, args.frame_size, args.frame_size),
            patch_size=(2, 16, 16),
            in_channels=3,
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            drop_rate=0.3,
            attn_drop_rate=0.2,
            pos_drop_rate=0.1,
            classifier='token'
        )
    
    if hasattr(args, 'use_class_weights') and args.use_class_weights:
        # Compute class frequencies from training dataset
        all_labels = [label for _, label in train_dataset]
        class_counts = torch.tensor(
            [count for _, count in sorted(Counter(all_labels).items())],
            dtype=torch.float32
        )
    
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = class_weights.to(device)
    
        print("Using class weights:", class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-4  # L2 regularization
    )
    
    # Scheduler: OneCycle with warm-up and cosine decay
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.2,  # 20% warm-up
        anneal_strategy='cos',  # Cosine annealing for smooth decay
        div_factor=25.0,  # Initial LR = max_lr / div_factor
        final_div_factor=1e4,  # Final LR = initial LR / final_div_factor
    )
        
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if args.use_amp else None
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Track best performance and starting epoch
    best_val_acc = 0.0
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        # Load checkpoint to CPU first to avoid GPU memory issues
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        # Create model and move to device first
        model = model.to(device)
        
        # Load model state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state dict and move to correct device
        if 'optimizer_state_dict' in checkpoint:
            optimizer_state = checkpoint['optimizer_state_dict']
            # Move optimizer state to GPU
            for state in optimizer_state['state'].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            optimizer.load_state_dict(optimizer_state)
        
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
    print(f"Resuming from epoch {start_epoch} with best val acc: {best_val_acc:.4f}")
    
    # Move model to device
    model.to(device)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        if args.dynamic_unfreeze and epoch == 2:
            print("Unfreezing backbone from epoch 3")
            for param in model.backbone.parameters():
                param.requires_grad = True
            
        # Training phase
        model.train()
        train_loss = 0.0
        train_corrects = 0
        total_samples = 0
        
        epoch_start = time.time()
        
        for i, (videos, labels) in enumerate(train_loader):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Zero gradients
            optimizer.zero_grad()
            
            if args.use_amp:
                # Mixed precision training
                with autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                
                # Backward pass with scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision
                outputs = model(videos)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Collect statistics
            with torch.no_grad():
                _, preds = torch.max(outputs, 1)
                batch_loss = loss.item()
                train_loss += batch_loss * videos.size(0)
                train_corrects += torch.sum(preds == labels.data)
                total_samples += videos.size(0)
            
            # Print progress every few batches
            if i % 5 == 0:
                batch_acc = torch.sum(preds == labels.data).double() / videos.size(0)
                print(f'Epoch {epoch+1}/{args.epochs} | Batch {i}/{len(train_loader)} | Loss: {batch_loss:.4f} | Acc: {batch_acc:.4f}')
        
        # Calculate epoch stats
        epoch_train_loss = train_loss / total_samples
        epoch_train_acc = train_corrects.double() / total_samples
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                if args.use_amp:
                    with autocast():
                        outputs = model(videos)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * videos.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_samples += videos.size(0)
        
        epoch_val_loss = val_loss / val_samples
        epoch_val_acc = val_corrects.double() / val_samples
        
        epoch_time = time.time() - epoch_start
        
        print(f'Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.1f}s')
        print(f'Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # save model checkpoint after every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_acc,
            'val_loss': epoch_val_loss, 
            'val_acc': epoch_val_acc,
            'num_classes': num_classes,
            'class_mapping': action_to_idx,
            'args': vars(args),
            'best_val_acc': best_val_acc,
        }
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_path = checkpoint_dir / f"model_best_{args.model_type}.pth"
            torch.save(checkpoint, best_path)
            print(f"✅ New best model saved with val_acc: {epoch_val_acc:.4f}")
        
        # Save periodic checkpoints
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            epoch_path = checkpoint_dir / f"model_epoch_{args.model_type}_{epoch+1}.pth"
            torch.save(checkpoint, epoch_path)
            print(f"📦 Checkpoint saved for epoch {epoch+1}")
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f} seconds")
    print(f"Final validation accuracy: {epoch_val_acc:.4f}")
    
    # Save quick test model
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_acc': epoch_val_acc,
        'num_classes': num_classes,
        'class_mapping': action_to_idx,  # Fix this line
    }, 'quick_test_model_15.pth')

    print(f"Model saved to quick_test_model_15_{args.model_type}.pth")
    
    # Determine feasibility
    if epoch_val_acc > 0.4:  # 40% accuracy is quite good for a quick test
        print("\n✅ FEASIBILITY TEST PASSED: Model is learning effectively!")
        print("You can proceed with full training on the complete dataset")
    elif epoch_val_acc > 0.2:  # Better than random for 10 classes
        print("\n🟨 FEASIBILITY TEST PARTIAL: Model is learning but could be improved")
        print("Consider adjusting model architecture or training parameters")
    else:
        print("\n❌ FEASIBILITY TEST FAILED: Model is not learning effectively")
        print("Consider significant changes to approach or data processing")


if __name__ == "__main__":
    print(f"Quick feasibility test with {args.subset:.1%} of data and top {args.top_n} classes")
    print(f"Configuration: {args}")
    train_quick_test(args)

    