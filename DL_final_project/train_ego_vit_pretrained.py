import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class PretrainedVideoViT(nn.Module):
    def __init__(self, num_classes, num_frames=8, freeze_backbone=True, dropout_rate=0.2):
        super().__init__()
        # Load pretrained ViT
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Freeze backbone if needed
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Add dropout layers
        self.feature_dropout = nn.Dropout(dropout_rate)
        
        # Add temporal modeling with attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.vit.config.hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=dropout_rate  # Add dropout to attention
        )
        
        # Add the missing dropout layer
        self.temporal_attention_dropout = nn.Dropout(dropout_rate)
        
        # Add classification head with dropout
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.vit.config.hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(self.vit.config.hidden_size, num_classes)
        )
        
        self.num_frames = num_frames
        
    def forward(self, x):  # x: [B, C, T, H, W]
        b, c, t, h, w = x.shape
        
        # Process each frame with ViT
        frame_features = []
        for i in range(t):
            # Extract frame
            frame = x[:, :, i]  # [B, C, H, W]
            
            # Resize to 224x224 (ViT expected size)
            if h != 224 or w != 224:
                frame = nn.functional.interpolate(frame, size=(224, 224), mode='bilinear', align_corners=False)
            
            # The HuggingFace ViT expects pixel_values as input
            outputs = self.vit(pixel_values=frame)
            
            # Extract CLS token features
            cls_token = outputs.last_hidden_state[:, 0]  # [B, D]
            frame_features.append(cls_token)
        
        # Stack frame features
        temporal_features = torch.stack(frame_features, dim=1)  # [B, T, D]
        
        # Apply temporal attention
        attn_output, _ = self.temporal_attention(
            temporal_features, temporal_features, temporal_features
        )
        attn_output = self.temporal_attention_dropout(attn_output)
        
        # Global temporal pooling (mean)
        pooled_features = torch.mean(attn_output, dim=1)  # [B, D]
        
        # Classification
        logits = self.classifier(pooled_features)
        
        return logits
    
class TemporalConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim)
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        # x: [B, T, D]
        batch_size, seq_len, feat_dim = x.shape
        
        # Apply temporal convolution
        x_t = x.transpose(1, 2)  # [B, D, T]
        conv_out = self.conv1d(x_t)  # [B, hidden_dim, T]
        conv_out = conv_out.transpose(1, 2)  # [B, T, hidden_dim]
        
        # Apply GRU
        gru_out, _ = self.gru(conv_out)  # [B, T, hidden_dim]
        return gru_out

# Use this in your model:
class ImprovedVideoViT(nn.Module):
    def __init__(self, num_classes, num_frames=8, freeze_backbone=True, dropout_rate=0.5):
        super().__init__()
        # ViT backbone
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        hidden_dim = self.vit.config.hidden_size
        
        # Freeze backbone initially
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
                
        # Regularization
        self.feature_dropout = nn.Dropout(dropout_rate * 0.6)
        
        # Improved temporal module
        self.temporal_model = TemporalConvGRU(hidden_dim, hidden_dim)
        self.temporal_dropout = nn.Dropout(dropout_rate)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.num_frames = num_frames
        
    # unfreeze_last_n_layers method same as before
        
    def forward(self, x):
        # Extract frame features same as before
        b, c, t, h, w = x.shape
    
        # Process each frame with ViT
        frame_features = []
        for i in range(t):
            # Extract frame
            frame = x[:, :, i, :, :]  # [B, C, H, W]
            
            # Resize to 224x224 (ViT expected size)
            if h != 224 or w != 224:
                frame = nn.functional.interpolate(frame, size=(224, 224), mode='bilinear', align_corners=False)
            
            # IMPORTANT: The HuggingFace ViT expects inputs as a dictionary
            # Do NOT use pixel_values as a named parameter, pass frame directly
            outputs = self.vit(frame)
            
            # Extract CLS token features with dropout
            cls_token = outputs.last_hidden_state[:, 0]  # [B, D]
            cls_token = self.feature_dropout(cls_token)  # Apply dropout to features
            frame_features.append(cls_token)
            
        # Then use the new temporal model:
        temporal_features = torch.stack(frame_features, dim=1)  # [B, T, D]
        temporal_features = self.temporal_dropout(temporal_features)
        
        # Process with temporal module
        temporal_output = self.temporal_model(temporal_features)  # [B, T, D]
        
        # Take the final timestep's output for classification
        final_features = temporal_output[:, -1, :]  # [B, D]
        
        # Classification
        logits = self.classifier(final_features)
        
        return logits