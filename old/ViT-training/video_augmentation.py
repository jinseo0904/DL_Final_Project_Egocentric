import torch
import random
import numpy as np
from torchvision import transforms

class RandomTemporalShift:
    """Randomly shift frames forward or backward in time."""
    def __init__(self, max_frames=2, p=0.5):
        self.max_frames = max_frames
        self.p = p
    
    def __call__(self, x):
        if random.random() < self.p:
            # x shape: [C, T, H, W]
            shift = random.randint(-self.max_frames, self.max_frames)
            if shift == 0:
                return x
            
            C, T, H, W = x.shape
            result = torch.zeros_like(x)
            
            if shift > 0:  # Shift right (forward in time)
                result[:, shift:, :, :] = x[:, :(T-shift), :, :]
                result[:, :shift, :, :] = x[:, 0:1, :, :].repeat(1, shift, 1, 1)  # Repeat first frame
            else:  # Shift left (backward in time)
                shift = abs(shift)
                result[:, :(T-shift), :, :] = x[:, shift:, :, :]
                result[:, (T-shift):, :, :] = x[:, -1:, :, :].repeat(1, shift, 1, 1)  # Repeat last frame
                
            return result
        return x

class RandomTemporalCrop:
    """Randomly crop a segment in time and rescale it."""
    def __init__(self, scale=(0.8, 1.0), p=0.5):
        self.scale = scale
        self.p = p
    
    def __call__(self, x):
        if random.random() < self.p:
            # x shape: [C, T, H, W]
            C, T, H, W = x.shape
            
            # Determine crop size
            crop_size = int(T * random.uniform(self.scale[0], self.scale[1]))
            crop_size = max(crop_size, 1)  # Ensure at least 1 frame
            
            # Determine start position
            start = random.randint(0, T - crop_size) if T > crop_size else 0
            
            # Crop the sequence
            cropped = x[:, start:start+crop_size, :, :]
            
            # Resize back to original length
            # This requires interpolation, different approaches possible
            if crop_size < T:
                # Simple approach: repeat the last frame
                result = torch.zeros_like(x)
                result[:, :crop_size, :, :] = cropped
                if crop_size < T:
                    result[:, crop_size:, :, :] = cropped[:, -1:, :, :].repeat(1, T-crop_size, 1, 1)
                return result
            else:
                return cropped
        return x

def get_enhanced_video_transforms(frame_size):
    """Create enhanced augmentation pipeline for video."""
    return transforms.Compose([
        # Spatial augmentations
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(frame_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # Temporal augmentations (applied after converting to tensor)
        RandomTemporalShift(max_frames=2, p=0.5),
        RandomTemporalCrop(scale=(0.8, 1.0), p=0.3),
    ])