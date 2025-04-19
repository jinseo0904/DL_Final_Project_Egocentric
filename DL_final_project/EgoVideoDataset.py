import os
import cv2
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Dataset paths
train_csv = pd.read_csv('/home/kim.jinseo/DL_final_project/CharadesEgo/CharadesEgo_v1_train_only1st.csv')
test_csv = pd.read_csv( '/home/kim.jinseo/DL_final_project/CharadesEgo/CharadesEgo_v1_test_only1st.csv')
VIDEO_PATH = '/scratch/kim.jinseo/CharadesEgo_v1_480'
CLASS_MAP = '/home/kim.jinseo/DL_final_project/CharadesEgo/Charades_v1_classes.txt'

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
                 action_to_idx=None, augment=True, samples_per_segment=1, 
                 window_size=2.0, window_stride=1.0):
        self.segments_df = segments_df
        self.video_path = video_path
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.action_to_idx = action_to_idx
        self.augment = augment
        self.samples_per_segment = samples_per_segment
        self.window_size = window_size  # Window size in seconds
        self.window_stride = window_stride  # Window stride in seconds
        
        # Pre-generate all windows for each segment to speed up __getitem__
        self.all_windows = self._generate_all_windows()
        
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
    
    def _generate_all_windows(self):
        """Generate all possible sliding windows for each segment"""
        all_windows = []
        for idx, row in self.segments_df.iterrows():
            video_id = row['video_id']
            segment_start = row['start']
            segment_end = row['end']
            action_class = row['action']
            
            segment_duration = segment_end - segment_start
            
            # For very short segments, just use the whole segment
            if segment_duration <= self.window_size:
                all_windows.append({
                    'video_id': video_id,
                    'window_start': segment_start,
                    'window_end': segment_end,
                    'action_class': action_class
                })
            else:
                # Generate overlapping windows
                num_windows = max(1, int((segment_duration - self.window_size) / self.window_stride) + 1)
                
                # If we need more windows than the stride would naturally create
                if num_windows < self.samples_per_segment and num_windows > 1:
                    # Adjust stride to get exactly samples_per_segment windows
                    adjusted_stride = (segment_duration - self.window_size) / (self.samples_per_segment - 1)
                    num_windows = self.samples_per_segment
                else:
                    adjusted_stride = self.window_stride
                
                for i in range(num_windows):
                    window_start = segment_start + i * adjusted_stride
                    window_end = min(window_start + self.window_size, segment_end)
                    
                    # Ensure we don't exceed the segment bounds
                    if window_end > segment_end:
                        window_start = segment_end - self.window_size
                        window_end = segment_end
                    
                    all_windows.append({
                        'video_id': video_id,
                        'window_start': window_start,
                        'window_end': window_end,
                        'action_class': action_class
                    })
        
        return all_windows
        
    def __len__(self):
        return len(self.all_windows)
    
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
        # Get the window information
        window = self.all_windows[idx]
        video_id = window['video_id']
        start_time = window['window_start']
        end_time = window['window_end']
        action_class = window['action_class']
        
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
        
        # Add some randomness for training
        if self.augment and np.random.random() < self.temporal_aug_prob:
            segment_length = end_frame - start_frame
            
            # Only apply temporal jitter if we have enough frames
            if segment_length > self.num_frames * 1.5:
                # Small random temporal jitter
                jitter_frames = int(segment_length * 0.1)  # 10% jitter
                start_frame = max(0, start_frame + np.random.randint(-jitter_frames, jitter_frames + 1))
                end_frame = min(total_frames, end_frame + np.random.randint(-jitter_frames, jitter_frames + 1))
        
        # Handle case where start and end frames are too close
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

def create_video_dataloaders(batch_size=2, num_frames=8, frame_size=(160, 160), num_workers=2,
                             window_size=2.0, window_stride=1.0):
    """
    Create dataloaders for training and validation with the appropriate parameters
    for the TimeSformer model, using sliding windows to create 2-second segments.
    """
    # Create datasets with the correct frame size and number of frames for the ViT model
    train_dataset = VideoSegmentDataset(
        train_df, 
        VIDEO_PATH, 
        num_frames=num_frames, 
        frame_size=frame_size, 
        action_to_idx=action_to_idx,
        window_size=window_size,
        window_stride=window_stride
    )
    
    val_dataset = VideoSegmentDataset(
        val_df, 
        VIDEO_PATH, 
        num_frames=num_frames, 
        frame_size=frame_size, 
        action_to_idx=action_to_idx,
        window_size=window_size,
        window_stride=window_size  # No overlap for validation
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

    print(f"Training samples with sliding windows: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Original segments: {len(train_df)}, Window size: {window_size}s, Stride: {window_stride}s")
    print(f"Number of batches in train_loader: {len(train_loader)}")

    return train_loader, val_loader

def create_video_dataloaders_top_n(batch_size=2, num_frames=8, frame_size=(160, 160), num_workers=2, 
                                   top_n_classes=20, window_size=2.0, window_stride=1.0):
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
        action_to_idx=simplified_action_to_idx,
        window_size=window_size,
        window_stride=window_stride
    )
    
    val_dataset = VideoSegmentDataset(
        val_df_filtered, 
        VIDEO_PATH, 
        num_frames=num_frames, 
        frame_size=frame_size, 
        action_to_idx=simplified_action_to_idx,
        window_size=window_size,
        window_stride=window_size  # No overlap for validation
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
    
    print(f"Simplified dataset with sliding windows - Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    print(f"Original segments: {len(train_df_filtered)}, Window size: {window_size}s, Stride: {window_stride}s")
    print(f"Using {len(simplified_action_to_idx)} classes instead of {len(action_to_idx)}")
    
    return train_loader, val_loader, simplified_action_to_idx

def create_video_dataloaders_subset(batch_size=4, num_frames=8, frame_size=(160, 160), 
                                    num_workers=2, top_n_classes=20, subset_fraction=0.1,
                                    window_size=2.0, window_stride=1.0):
    """
    Create dataloaders with only a subset of the data for quick testing
    
    Parameters:
    - batch_size: Batch size for training
    - num_frames: Number of frames to sample per video
    - frame_size: Size to resize frames to
    - num_workers: Number of workers for data loading
    - top_n_classes: Number of top classes to use (set to None to use all classes)
    - subset_fraction: Fraction of the dataset to use (0.1 = 10%)
    - window_size: Size of sliding window in seconds
    - window_stride: Stride between windows in seconds
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
        action_to_idx=simplified_action_to_idx,
        window_size=window_size,
        window_stride=window_stride
    )
    
    val_dataset = VideoSegmentDataset(
        val_df_subset, 
        VIDEO_PATH, 
        num_frames=num_frames, 
        frame_size=frame_size, 
        action_to_idx=simplified_action_to_idx,
        window_size=window_size,
        window_stride=window_size  # No overlap for validation
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
    
    print(f"Subset dataset with sliding windows - Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")
    print(f"Original segments: {len(train_df_subset)}, Window size: {window_size}s, Stride: {window_stride}s")
    print(f"Using {len(simplified_action_to_idx)} classes")
    print(f"Number of batches in training: {len(train_loader)}")
    
    return train_loader, val_loader, simplified_action_to_idx

def create_video_dataloaders_selected_classes(batch_size=4, num_frames=8, frame_size=(160, 160), 
                                            num_workers=2, class_file='selected_classes.txt',
                                            subset_fraction=1.0, use_augmentation=True,
                                            window_size=2.0, window_stride=1.0):
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
        augment=use_augmentation,  # Enable augmentation for training
        window_size=window_size,
        window_stride=window_stride
    )
    
    val_dataset = VideoSegmentDataset(
        val_df_filtered, 
        VIDEO_PATH, 
        num_frames=num_frames, 
        frame_size=frame_size, 
        action_to_idx=selected_action_to_idx,
        augment=False,  # No augmentation for validation
        window_size=window_size,
        window_stride=window_size  # No overlap for validation
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
    
    print(f"Selected classes dataset with sliding windows - Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    print(f"Original segments: {len(train_df_filtered)}, Window size: {window_size}s, Stride: {window_stride}s")
    print(f"Number of batches in training: {len(train_loader)}")
    
    return train_loader, val_loader, selected_action_to_idx


if __name__ == "__main__":
    # Test the dataloader with updated parameters for TimeSformer model
    train_loader, val_loader = create_video_dataloaders(
        batch_size=2,
        num_frames=8,  # Use 8 frames as specified for TimeSformer
        frame_size=(160, 160),
        num_workers=2,
        window_size=2.0,  # 2-second windows
        window_stride=1.0  # 1-second stride for 50% overlap
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