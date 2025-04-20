import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import argparse
from torch.cuda.amp import autocast, GradScaler

# Import our models and dataset
from video_vit import VideoViT, TransformerBlock
from EgoVideoDataset import create_video_dataloaders

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parse arguments
parser = argparse.ArgumentParser(description='Train VideoViT on EgoVideo dataset')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--num_frames', type=int, default=16, help='Number of frames')
parser.add_argument('--frame_size', type=int, default=160, help='Frame size')
parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
parser.add_argument('--depth', type=int, default=6, help='Transformer depth')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training')
parser.add_argument('--accum_steps', type=int, default=4, help='Gradient accumulation steps')
parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
args = parser.parse_args()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def enable_gradient_checkpointing():
    """
    Enable gradient checkpointing for transformer blocks to save memory
    """
    original_forward = TransformerBlock.forward
    
    def forward_with_checkpoint(self, x):
        if hasattr(self, 'checkpoint') and self.checkpoint:
            # Use torch.utils.checkpoint for memory efficiency
            from torch.utils.checkpoint import checkpoint
            return checkpoint(original_forward, self, x)
        else:
            return original_forward(self, x)
    
    # Patch the TransformerBlock's forward method
    TransformerBlock.forward = forward_with_checkpoint
    print("Gradient checkpointing enabled for transformer blocks")

class VideoViTWithCheckpointing(VideoViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Enable gradient checkpointing for transformer blocks
        for block in self.blocks:
            block.checkpoint = True

def train(args):
    # Create dataloaders
    train_loader, val_loader = create_video_dataloaders(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        frame_size=(args.frame_size, args.frame_size),
        num_workers=args.num_workers
    )
    
    # Get number of classes from the dataset
    num_classes = len(train_loader.dataset.action_to_idx)
    print(f"Training with {num_classes} classes")
    
    # Create model configuration
    video_size = (args.num_frames, args.frame_size, args.frame_size)
    patch_size = (4, 16, 16)  # Adjust patch size for memory efficiency
    
    # Initialize model with checkpointing for memory efficiency
    try:
        enable_gradient_checkpointing()
        model = VideoViTWithCheckpointing(
            video_size=video_size,
            patch_size=patch_size,
            in_channels=3,
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            pos_drop_rate=0.1,
            classifier='token'
        )
        print("Using model with gradient checkpointing")
    except Exception as e:
        print(f"Failed to enable checkpointing: {e}")
        model = VideoViT(
            video_size=video_size,
            patch_size=patch_size,
            in_channels=3,
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            pos_drop_rate=0.1,
            classifier='token'
        )
        print("Using standard model without gradient checkpointing")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if args.use_amp else None
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Move model to device
    model.to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_corrects = 0
        total_samples = 0
        
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        for i, (videos, labels) in enumerate(train_loader):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if args.use_amp:
                # Use mixed precision
                with autocast():
                    # Forward pass
                    outputs = model(videos)
                    loss = criterion(outputs, labels) / args.accum_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Update weights only after accumulating gradients
                if (i + 1) % args.accum_steps == 0 or (i + 1) == len(train_loader):
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Optimizer step with gradient scaling
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard precision training
                outputs = model(videos)
                loss = criterion(outputs, labels) / args.accum_steps
                loss.backward()
                
                # Update weights only after accumulating gradients
                if (i + 1) % args.accum_steps == 0 or (i + 1) == len(train_loader):
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Statistics
            with torch.no_grad():
                _, preds = torch.max(outputs, 1)
                batch_loss = loss.item() * args.accum_steps
                train_loss += batch_loss * videos.size(0)
                train_corrects += torch.sum(preds == labels.data)
                total_samples += videos.size(0)
            
            # Print progress every 10 batches
            if i % 10 == 0:
                print(f'Epoch {epoch+1}/{args.epochs} | Batch {i}/{len(train_loader)} | Loss: {batch_loss:.4f}')
            
            # Clear cache every few iterations
            if torch.cuda.is_available() and (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
        
        # Update learning rate
        scheduler.step()
        
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
                    # Use mixed precision for validation too
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
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.1f}s')
        print(f'Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            print(f'New best model with val loss: {best_val_loss:.4f}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
            }, 'best_ego_vit_model.pth')
        
        # Save regular checkpoint for resuming training
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': epoch_val_loss,
        }, 'checkpoint_ego_vit_model.pth')

if __name__ == "__main__":
    print("VideoViT training for EgoVideo dataset")
    print(f"Configuration: {args}")
    train(args) 