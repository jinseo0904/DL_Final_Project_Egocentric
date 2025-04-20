import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from video_vit import VideoViT

# Import your dataset class here
# from your_dataset_file import VideoSegmentDataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Memory efficiency settings
torch.backends.cudnn.benchmark = True

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Hyperparameters
batch_size = 2  # Reduced batch size for memory efficiency
num_epochs = 30
learning_rate = 3e-5  # Reduced learning rate for ViT
weight_decay = 1e-4  # Weight decay for regularization

# Model configuration - smaller and more memory efficient
video_size = (16, 160, 160)  # (T, H, W)
patch_size = (2, 16, 16)     # Larger patches = fewer patches = less memory
embed_dim = 384             # Reduced from standard 768
depth = 8                   # Reduced from standard 12
num_heads = 6               # Reduced from standard 12
num_classes = 157           # Your specific number of classes

# Data loading and processing
# train_dataset = VideoSegmentDataset(...)
# val_dataset = VideoSegmentDataset(...)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# Initialize model
model = VideoViT(
    video_size=video_size,
    patch_size=patch_size,
    in_channels=3,
    num_classes=num_classes,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    drop_rate=0.1,           # Dropout for regularization
    attn_drop_rate=0.1,     # Attention dropout
    pos_drop_rate=0.1,      # Position embedding dropout
    classifier='token'      # Use cls token for classification
)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Gradient accumulation for virtual larger batch sizes
gradient_accumulation_steps = 4  # Effectively increases batch size by 4x

# Training function with memory optimizations
def train_with_gradient_accumulation(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, gradient_accumulation_steps=1):
    model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_corrects = 0
        total_samples = 0
        
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        for i, (videos, labels) in enumerate(train_loader):
            videos = videos.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(videos)
            loss = criterion(outputs, labels) / gradient_accumulation_steps  # Scale loss
            
            # Backward pass
            loss.backward()
            
            # Update weights only after accumulating gradients
            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            batch_loss = loss.item() * gradient_accumulation_steps  # Scale back for reporting
            train_loss += batch_loss * videos.size(0)
            train_corrects += torch.sum(preds == labels.data)
            total_samples += videos.size(0)
            
            # Clear cache every few iterations
            if torch.cuda.is_available() and (i + 1) % 10 == 0:
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
                videos = videos.to(device)
                labels = labels.to(device)
                
                outputs = model(videos)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * videos.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_samples += videos.size(0)
        
        epoch_val_loss = val_loss / val_samples
        epoch_val_acc = val_corrects.double() / val_samples
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.1f}s')
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
            }, 'best_video_vit_model.pth')
    
    return model

# Uncomment to train the model
# train_with_gradient_accumulation(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, gradient_accumulation_steps)

# Function to load the best model
def load_best_model(model_path, model):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}")
    return model

# Example usage:
# model = load_best_model('best_video_vit_model.pth', model) 