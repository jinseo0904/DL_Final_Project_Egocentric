import torch
import torch.nn as nn
import numpy as np
import argparse
import pandas as pd
import os
import cv2
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from video_vit import VideoViT

# Parse arguments
parser = argparse.ArgumentParser(description='Test VideoViT on EgoVideo dataset')
parser.add_argument('--model_path', type=str, default='best_ego_vit_model.pth', help='Path to model checkpoint')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_frames', type=int, default=16, help='Number of frames')
parser.add_argument('--frame_size', type=int, default=160, help='Frame size')
parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
parser.add_argument('--use_amp', action='store_true', help='Use mixed precision')
args = parser.parse_args()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Paths
TEST_CSV_PATH = '/mnt/d/CharadesEgo/Annotations/CharadesEgo_v1_test_only1st.csv'
VIDEO_PATH = '/mnt/d/CharadesEgo/CharadesEgo_v1_480'
CLASS_MAP_PATH = '/mnt/d/CharadesEgo/Annotations/Charades_v1_classes.txt'

def prepare_test_data():
    """
    Prepare test data from the Charades-Ego dataset
    """
    # Read class map
    class_map = {}
    with open(CLASS_MAP_PATH, 'r') as f:
        for line in f:
            class_id, description = line.strip().split(' ', 1)
            class_map[class_id] = description
    
    # Create action to index mapping
    unique_actions = sorted(list(class_map.keys()))
    action_to_idx = {action: idx for idx, action in enumerate(unique_actions)}
    idx_to_action = {idx: action for action, idx in action_to_idx.items()}
    
    # Read and process test data
    test_csv = pd.read_csv(TEST_CSV_PATH)
    test_csv = test_csv[test_csv['actions'].apply(type) == str]
    
    def parse_action_segments(action_str):
        segments = []
        for segment in action_str.split(';'):
            # split segment by space
            segment_parts = segment.split(' ')
            # convert to tuple of str, float, float
            if len(segment_parts) >= 3:
                segments.append((segment_parts[0], float(segment_parts[1]), float(segment_parts[2])))
        return segments
    
    # Extract action segments
    test_segments = []
    for _, row in test_csv.iterrows():
        video_id = row['id']
        if isinstance(row['actions'], str):
            segments = parse_action_segments(row['actions'])
            for action, start, end in segments:
                test_segments.append({
                    'video_id': video_id,
                    'action': action,
                    'start': start,
                    'end': end
                })
    
    # Convert to DataFrame
    test_df = pd.DataFrame(test_segments)
    if len(test_df) > 0:
        test_df['action_duration'] = test_df['end'] - test_df['start']
    
    return test_df, action_to_idx, idx_to_action, class_map

class TestVideoDataset(Dataset):
    def __init__(self, segments_df, video_path, num_frames=16, frame_size=(160, 160), action_to_idx=None):
        self.segments_df = segments_df
        self.video_path = video_path
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.action_to_idx = action_to_idx
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.segments_df)
    
    def __getitem__(self, idx):
        # Get video info
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
        
        # Handle case where start and end frames are the same
        if end_frame <= start_frame:
            # If segment is too short, use a single frame and repeat it
            frame_indices = np.full(self.num_frames, start_frame)
        else:
            # Sample frames evenly from the segment
            if end_frame - start_frame >= self.num_frames:
                # If we have more frames than needed, sample evenly
                frame_indices = np.linspace(start_frame, end_frame-1, self.num_frames, dtype=int)
            else:
                # If we have fewer frames, loop the video
                frame_indices = np.array([start_frame + (i % (end_frame - start_frame)) for i in range(self.num_frames)])
        
        # Extract frames
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                # If frame reading fails, create a blank frame
                frame = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
            frames.append(frame)
        
        cap.release()
        
        # For VideoViT: Stack frames into shape [C, T, H, W]
        video_tensor = torch.stack(frames)  # Shape: [T, C, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # Shape: [C, T, H, W]
        
        return video_tensor, action_idx, video_id

def test_model(args):
    """
    Test the trained VideoViT model on the test dataset
    """
    # Prepare test data
    test_df, action_to_idx, idx_to_action, class_map = prepare_test_data()
    num_classes = len(action_to_idx)
    
    # Create test dataset and dataloader
    test_dataset = TestVideoDataset(
        test_df, 
        VIDEO_PATH, 
        num_frames=args.num_frames, 
        frame_size=(args.frame_size, args.frame_size), 
        action_to_idx=action_to_idx
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load model
    video_size = (args.num_frames, args.frame_size, args.frame_size)
    patch_size = (4, 16, 16)  # Same as in training
    
    model = VideoViT(
        video_size=video_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=256,  # Same as in training
        depth=6,        # Same as in training
        num_heads=4,    # Same as in training
        classifier='token'
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')} with validation loss {checkpoint.get('val_loss', 'unknown')}")
    
    # Evaluate model
    all_preds = []
    all_labels = []
    all_video_ids = []
    
    with torch.no_grad():
        for videos, labels, video_ids in tqdm(test_loader, desc="Testing"):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if args.use_amp:
                with autocast():
                    outputs = model(videos)
            else:
                outputs = model(videos)
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_video_ids.extend(video_ids)
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Generate classification report
    class_names = [class_map[idx_to_action[i]] for i in range(num_classes)]
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print("Classification Report:")
    print(report)
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'video_id': all_video_ids,
        'true_label': [idx_to_action[label] for label in all_labels],
        'predicted_label': [idx_to_action[pred] for pred in all_preds],
        'correct': np.array(all_preds) == np.array(all_labels)
    })
    
    results_df.to_csv('test_results.csv', index=False)
    print("Results saved to test_results.csv")
    
    # Return metrics for further analysis
    return accuracy, all_labels, all_preds, idx_to_action, class_map

if __name__ == "__main__":
    print("Testing VideoViT model on EgoVideo dataset")
    print(f"Configuration: {args}")
    accuracy, labels, predictions, idx_to_action, class_map = test_model(args) 