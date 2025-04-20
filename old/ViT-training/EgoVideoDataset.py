import os
import cv2
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Dataset paths
annotation_path = 'data/CharadesEgo/'
print(os.listdir(annotation_path))
train_csv = pd.read_csv('data/CharadesEgo/CharadesEgo_v1_train_only1st.csv')
test_csv = pd.read_csv('data/CharadesEgo/CharadesEgo_v1_test_only1st.csv')
VIDEO_PATH = 'data/CharadesEgo/CharadesEgo_v1_480'
CLASS_MAP = 'data/CharadesEgo/Charades_v1_classes.txt'

# make sure the full content of each cell is visible
pd.set_option('display.max_colwidth', None)

# read CLASS_MAP into a dictionary mapping class IDs to descriptions
class_map = {}
with open(CLASS_MAP, 'r') as f:
    for line in f:
        class_id, description = line.strip().split(' ', 1)
        class_map[class_id] = description
print("First 10 classes:")
for i, (class_id, desc) in enumerate(class_map.items()):
    if i >= 10:
        break
    print(f"{class_id}: {desc}")

def parse_action_segments(action_segments):
    segments = []
    for segment in action_segments:
        # split segment by space
        segment = segment.split(' ')
        # convert to tuple of str, float, float
        segments.append((segment[0], float(segment[1]), float(segment[2])))
    return segments

# Filter and process training data
print(f"Initial training data rows: {len(train_csv)}")
train_csv = train_csv[train_csv['actions'].apply(type) == str]
print(f"After filtering rows: {len(train_csv)}")

# Convert actions column to list and parse segments
train_csv['actions_list'] = train_csv['actions'].apply(lambda x: x.split(';'))
train_csv['actions_list'] = train_csv['actions_list'].apply(parse_action_segments)

# Merge all action_list values into a single list with video IDs
all_action_segments = [(segment, id) for id, action_list in zip(train_csv['id'], train_csv['actions_list']) for segment in action_list]
print(f"Total action segments: {len(all_action_segments)}")

# Convert to DataFrame
action_segments_df = pd.DataFrame([(action, start, end, video_id) for (action, start, end), video_id in all_action_segments], 
                                columns=['action', 'start', 'end', 'video_id'])

# Add column for video duration
action_segments_df['action_duration'] = action_segments_df['end'] - action_segments_df['start']
print(action_segments_df.head())

# Map action classes to integers
unique_actions = sorted(list(class_map.keys()))
action_to_idx = {action: idx for idx, action in enumerate(unique_actions)}
idx_to_action = {idx: action for action, idx in action_to_idx.items()}

print(f"Number of unique action classes: {len(unique_actions)}")
print("\nFirst few action mappings:")
for i in range(min(5, len(unique_actions))):
    print(f"{unique_actions[i]} -> {action_to_idx[unique_actions[i]]}")

# Split data into training and validation sets
train_df, val_df = train_test_split(action_segments_df, test_size=0.2, random_state=42, stratify=action_segments_df['action'])
print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")

class VideoSegmentDataset(Dataset):
    def __init__(self, segments_df, video_path, num_frames=16, frame_size=(160, 160), 
                 action_to_idx=None, augment=True):
        self.segments_df = segments_df
        self.video_path = video_path
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.action_to_idx = action_to_idx
        self.augment = augment
        
        # Define separate transforms for training (with augmentation) and validation/testing
        if self.augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((int(frame_size[0]*1.2), int(frame_size[1]*1.2))),  # Resize larger
                transforms.RandomCrop(frame_size),  # Then random crop
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.1
                ),  # Color augmentation
                transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip with 50% probability
                transforms.RandomAffine(
                    degrees=10,  # Rotate +/- 10 degrees
                    translate=(0.1, 0.1),  # Translate by up to 10% in each direction
                    scale=(0.9, 1.1),  # Scale by 90% to 110%
                ),  # Geometric augmentation
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2)  # Randomly erase rectangles from the image
            ])
        else:
            # Simple transform for validation/testing
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(frame_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        # Temporal augmentation parameters
        self.temporal_aug_prob = 0.5 if augment else 0  # Probability of applying temporal augmentation
        
    def __len__(self):
        return len(self.segments_df)
    
    def _safe_read_frame(self, cap, frame_idx):
        """Safely read a frame from video with error handling"""
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                # Create a blank frame if reading fails
                return np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
            
            # Convert BGR to RGB
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            # Catch any errors and return a blank frame
            return np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
    
    def __getitem__(self, idx):
        # Get video info (same as before)
        video_id = self.segments_df.iloc[idx]['video_id']
        start_time = self.segments_df.iloc[idx]['start']
        end_time = self.segments_df.iloc[idx]['end']
        action_class = self.segments_df.iloc[idx]['action']
        
        # Convert action class to integer index
        if self.action_to_idx is not None:
            action_idx = self.action_to_idx[action_class]
        else:
            action_idx = action_class
        
        # Load video
        video_file = os.path.join(self.video_path, video_id + '.mp4')
        cap = cv2.VideoCapture(video_file)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate start and end frames
        start_frame = int(start_time * fps)
        end_frame = min(int(end_time * fps), total_frames)
        
        # ===== NEW: TEMPORAL AUGMENTATION =====
        if self.augment and np.random.random() < self.temporal_aug_prob:
            # Apply various temporal augmentations
            segment_length = end_frame - start_frame
            
            if segment_length > self.num_frames * 2:  # Only apply if segment is long enough
                # 1. Temporal cropping - take a smaller part of the segment
                crop_size = max(self.num_frames, int(segment_length * np.random.uniform(0.5, 0.9)))
                offset = np.random.randint(0, segment_length - crop_size + 1)
                start_frame = start_frame + offset
                end_frame = start_frame + crop_size
                
                # 2. Random temporal shift - shift the whole segment slightly
                shift_amount = int(segment_length * np.random.uniform(-0.1, 0.1))
                if shift_amount != 0:
                    start_frame = max(0, start_frame + shift_amount)
                    end_frame = min(total_frames, end_frame + shift_amount)
        
        # Handle case where start and end frames are too close (same as before but with improved comments)
        if end_frame <= start_frame + 1:  # Need at least 2 frames
            # If segment is too short, use the start frame and repeat it
            frame_indices = np.full(self.num_frames, start_frame)
        else:
            # Sample frames from the segment
            if end_frame - start_frame >= self.num_frames:
                # If we have enough frames, sample evenly or randomly
                if self.augment and np.random.random() < 0.3:
                    # Random sampling - take frames with random intervals
                    frame_indices = sorted(np.random.choice(
                        range(start_frame, end_frame), 
                        size=self.num_frames, 
                        replace=False
                    ))
                else:
                    # Even sampling - take frames at regular intervals
                    frame_indices = np.linspace(start_frame, end_frame-1, self.num_frames, dtype=int)
            else:
                # If we don't have enough frames, loop the available ones
                frame_indices = np.array([
                    start_frame + (i % (end_frame - start_frame)) 
                    for i in range(self.num_frames)
                ])
        
        # Extract frames with robust error handling
        frames = []
        for frame_idx in frame_indices:
            # Use our safe frame reading method
            frame = self._safe_read_frame(cap, frame_idx)
            
            # Apply transformation - this handles conversion to tensor
            try:
                frame_tensor = self.transform(frame)
                frames.append(frame_tensor)
            except Exception:
                # If transformation fails, create a blank tensor of the right shape
                # Calculate expected shape based on other successful frames or default size
                if frames:
                    # Copy the shape of previous frame
                    blank_tensor = torch.zeros_like(frames[-1])
                else:
                    # Create default shape
                    c = 3  # RGB channels
                    h, w = self.frame_size
                    blank_tensor = torch.zeros((c, h, w))
                    # Apply normalization to match what transform would do
                    blank_tensor = torch.nn.functional.normalize(blank_tensor, dim=0)
                
                frames.append(blank_tensor)

        cap.release()
        
        # Stack frames into shape [C, T, H, W]
        video_tensor = torch.stack(frames)  # Shape: [T, C, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # Shape: [C, T, H, W]
        
        # ===== CONSISTENCY AUGMENTATION =====
        # Make all frames consistent with each other
        if self.augment and np.random.random() < 0.3:
            # Apply a consistent color shift across all frames
            c, t, h, w = video_tensor.shape
            color_shift = torch.randn(c, 1, 1, 1, device=video_tensor.device) * 0.05
            video_tensor = video_tensor + color_shift
        
        return video_tensor, action_idx

def create_video_dataloaders(batch_size=2, num_frames=16, frame_size=(160, 160), num_workers=2):
    """
    Create dataloaders for training and validation with the appropriate parameters
    for the VideoViT model.
    """
    # Create datasets with the correct frame size and number of frames for the ViT model
    train_dataset = VideoSegmentDataset(
        train_df, 
        VIDEO_PATH, 
        num_frames=num_frames, 
        frame_size=frame_size, 
        action_to_idx=action_to_idx
    )
    
    val_dataset = VideoSegmentDataset(
        val_df, 
        VIDEO_PATH, 
        num_frames=num_frames, 
        frame_size=frame_size, 
        action_to_idx=action_to_idx
    )
    
    # Create DataLoaders with memory-efficient settings
    train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last batch if incomplete (helps with batch norm)
    )
        
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
            num_workers=num_workers,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Number of batches in train_loader: {len(train_loader)}")

    return train_loader, val_loader

def create_video_dataloaders_top_n(batch_size=2, num_frames=16, frame_size=(160, 160), num_workers=2, top_n_classes=20):
    """Create dataloaders with only the top N most frequent action classes"""
    
    # Get class counts in the original dataframe
    class_counts = action_segments_df['action'].value_counts()
    top_classes = class_counts.nlargest(top_n_classes).index.tolist()
    
    # Filter dataframes
    train_df_filtered = train_df[train_df['action'].isin(top_classes)]
    val_df_filtered = val_df[val_df['action'].isin(top_classes)]
    
    # Create new action mapping for just these classes
    simplified_action_to_idx = {action: idx for idx, action in enumerate(sorted(top_classes))}
    
    # Print the selected classes
    print(f"\nUsing top {top_n_classes} classes:")
    for cls in sorted(top_classes):
        count = class_counts[cls]
        print(f"  {cls}: {class_map[cls]} ({count} samples)")
    
    # Create datasets using the simplified mapping
    train_dataset = VideoSegmentDataset(
        train_df_filtered, 
        VIDEO_PATH, 
        num_frames=num_frames, 
        frame_size=frame_size, 
        action_to_idx=simplified_action_to_idx
    )
    
    val_dataset = VideoSegmentDataset(
        val_df_filtered, 
        VIDEO_PATH, 
        num_frames=num_frames, 
        frame_size=frame_size, 
        action_to_idx=simplified_action_to_idx
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Simplified dataset - Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    print(f"Using {len(simplified_action_to_idx)} classes instead of {len(action_to_idx)}")
    
    return train_loader, val_loader, simplified_action_to_idx

def create_video_dataloaders_subset(batch_size=4, num_frames=16, frame_size=(160, 160), 
                                    num_workers=2, top_n_classes=20, subset_fraction=0.1):
    """
    Create dataloaders with only a subset of the data for quick testing
    
    Parameters:
    - batch_size: Batch size for training
    - num_frames: Number of frames to sample per video
    - frame_size: Size to resize frames to
    - num_workers: Number of workers for data loading
    - top_n_classes: Number of top classes to use (set to None to use all classes)
    - subset_fraction: Fraction of the dataset to use (0.1 = 10%)
    """
    # First determine what classes to use
    if top_n_classes is not None:
        # Get class counts in the original dataframe
        class_counts = action_segments_df['action'].value_counts()
        top_classes = class_counts.nlargest(top_n_classes).index.tolist()
        
        # Filter dataframes to only include these classes
        train_df_filtered = train_df[train_df['action'].isin(top_classes)]
        val_df_filtered = val_df[val_df['action'].isin(top_classes)]
        
        # Create new action mapping for just these classes
        simplified_action_to_idx = {action: idx for idx, action in enumerate(sorted(top_classes))}
        
        print(f"\nUsing top {top_n_classes} classes:")
        for cls in sorted(top_classes):
            count = class_counts[cls]
            print(f"  {cls}: {class_map[cls]} ({count} samples)")
    else:
        # Use all classes
        train_df_filtered = train_df
        val_df_filtered = val_df
        simplified_action_to_idx = action_to_idx
    
    # Now take a subset of the data
    if subset_fraction < 1.0:
        # Ensure balanced sampling across classes
        train_subset = []
        for cls in simplified_action_to_idx.keys():
            # Get all samples for this class
            cls_samples = train_df_filtered[train_df_filtered['action'] == cls]
            # Take a subset
            subset_size = max(int(len(cls_samples) * subset_fraction), 1)  # At least 1 sample
            cls_subset = cls_samples.sample(n=subset_size, random_state=42)
            train_subset.append(cls_subset)
        
        # Combine all sampled subsets
        train_df_subset = pd.concat(train_subset)
        
        # Do the same for validation set but with more samples to get reliable metrics
        val_subset = []
        val_fraction = min(subset_fraction * 2, 0.5)  # More validation samples, but max 50%
        for cls in simplified_action_to_idx.keys():
            cls_samples = val_df_filtered[val_df_filtered['action'] == cls]
            subset_size = max(int(len(cls_samples) * val_fraction), 1)  # At least 1 sample
            cls_subset = cls_samples.sample(n=subset_size, random_state=42)
            val_subset.append(cls_subset)
        
        val_df_subset = pd.concat(val_subset)
        
        print(f"Using {subset_fraction:.1%} of training data and {val_fraction:.1%} of validation data")
    else:
        train_df_subset = train_df_filtered
        val_df_subset = val_df_filtered
    
    # Create datasets
    train_dataset = VideoSegmentDataset(
        train_df_subset, 
        VIDEO_PATH, 
        num_frames=num_frames, 
        frame_size=frame_size, 
        action_to_idx=simplified_action_to_idx
    )
    
    val_dataset = VideoSegmentDataset(
        val_df_subset, 
        VIDEO_PATH, 
        num_frames=num_frames, 
        frame_size=frame_size, 
        action_to_idx=simplified_action_to_idx
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Subset dataset - Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")
    print(f"Using {len(simplified_action_to_idx)} classes")
    print(f"Number of batches in training: {len(train_loader)}")
    
    return train_loader, val_loader, simplified_action_to_idx

def create_video_dataloaders_selected_classes(batch_size=4, num_frames=16, frame_size=(160, 160), 
                                            num_workers=2, class_file='selected_classes.txt',
                                            subset_fraction=1.0, use_augmentation=True):
    """
    Create dataloaders with only the selected classes from the text file
    """
    # Load selected classes from file
    selected_classes = []
    with open(class_file, 'r') as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith('#') or line.strip() == '':
                continue
            # Parse class ID from CSV format
            class_id = line.strip().split(',')[0]
            selected_classes.append(class_id)
    
    print(f"Loaded {len(selected_classes)} selected classes from {class_file}")
    
    # Filter dataframes to only include these classes
    train_df_filtered = train_df[train_df['action'].isin(selected_classes)]
    val_df_filtered = val_df[val_df['action'].isin(selected_classes)]
    
    # Apply subset filtering if requested
    if subset_fraction < 1.0:
        print(f"Using {subset_fraction:.1%} of the data")
        # Ensure balanced sampling across classes
        train_subset = []
        for cls in selected_classes:
            # Get all samples for this class
            cls_samples = train_df_filtered[train_df_filtered['action'] == cls]
            # Take a subset
            subset_size = max(int(len(cls_samples) * subset_fraction), 1)  # At least 1 sample
            cls_subset = cls_samples.sample(n=subset_size, random_state=42)
            train_subset.append(cls_subset)
        
        # Combine all sampled subsets
        train_df_filtered = pd.concat(train_subset)
        
        # Do the same for validation set
        val_subset = []
        val_fraction = min(subset_fraction * 2, 0.5)  # More validation samples, but max 50%
        for cls in selected_classes:
            cls_samples = val_df_filtered[val_df_filtered['action'] == cls]
            subset_size = max(int(len(cls_samples) * val_fraction), 1)  # At least 1 sample
            cls_subset = cls_samples.sample(n=subset_size, random_state=42)
            val_subset.append(cls_subset)
        
        val_df_filtered = pd.concat(val_subset)
    
    # Create new action mapping for just these classes
    selected_action_to_idx = {action: idx for idx, action in enumerate(sorted(selected_classes))}
    
    print(f"\nUsing selected diverse classes:")
    for cls in sorted(selected_classes):
        count = len(train_df_filtered[train_df_filtered['action'] == cls])
        print(f"  {cls}: {class_map[cls]} ({count} samples)")
    
    # Create datasets using the selected mapping
    train_dataset = VideoSegmentDataset(
        train_df_filtered, 
        VIDEO_PATH, 
        num_frames=num_frames, 
        frame_size=frame_size, 
        action_to_idx=selected_action_to_idx,
        augment=use_augmentation  # Enable augmentation for training
    )
    
    val_dataset = VideoSegmentDataset(
        val_df_filtered, 
        VIDEO_PATH, 
        num_frames=num_frames, 
        frame_size=frame_size, 
        action_to_idx=selected_action_to_idx,
        augment=False  # No augmentation for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Selected classes dataset - Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    print(f"Number of batches in training: {len(train_loader)}")
    
    return train_loader, val_loader, selected_action_to_idx


if __name__ == "__main__":
    # Test the dataloader with sample parameters for VideoViT
    train_loader, val_loader = create_video_dataloaders(
        batch_size=2,
        num_frames=16,  # Match the number of frames expected by the model
        frame_size=(160, 160),  # Match the frame size expected by the model
        num_workers=2
    )
    
    # Verify the shape of the tensors
    sample_batch = next(iter(train_loader))
    videos, labels = sample_batch
    print(f"\nSample batch video tensor shape: {videos.shape}")  # Should be [B, C, T, H, W]
    print(f"Sample batch labels shape: {labels.shape}")
    print(f"Sample labels: {labels}")
    print(f"Label types: {labels.dtype}")
    
    # Calculate and print number of classes for model initialization
    num_classes = len(action_to_idx)
    print(f"Number of classes for model: {num_classes}")