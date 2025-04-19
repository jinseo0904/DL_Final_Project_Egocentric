import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

class PatchEmbedding(nn.Module):
    """
    Split video into patches and then embed them.
    """
    def __init__(self, video_size=(16, 224, 224), patch_size=(2, 16, 16), in_channels=3, embed_dim=768):
        super().__init__()
        self.video_size = video_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.num_patches = (video_size[0] // patch_size[0]) * (video_size[1] // patch_size[1]) * (video_size[2] // patch_size[2])
        self.proj = nn.Conv3d(in_channels, embed_dim, 
                            kernel_size=patch_size, 
                            stride=patch_size)
    
    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        assert T == self.video_size[0] and H == self.video_size[1] and W == self.video_size[2], \
            f"Input video size ({T}, {H}, {W}) doesn't match model's expected size {self.video_size}"
        
        # (B, C, T, H, W) -> (B, E, T', H', W')
        x = self.proj(x)
        # (B, E, T', H', W') -> (B, T'*H'*W', E)
        x = rearrange(x, 'b e t h w -> b (t h w) e')
        
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D/H)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VideoViT(nn.Module):
    """
    Vision Transformer for Video Classification
    """
    def __init__(self, 
                 video_size=(16, 160, 160), 
                 patch_size=(2, 16, 16), 
                 in_channels=3, 
                 num_classes=157,
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 pos_drop_rate=0.,
                 classifier='token'):
        super().__init__()
        
        self.num_classes = num_classes
        self.classifier = classifier  # 'token' or 'gap'
        
        self.patch_embed = PatchEmbedding(
            video_size=video_size, 
            patch_size=patch_size, 
            in_channels=in_channels, 
            embed_dim=embed_dim
        )
        
        self.num_patches = self.patch_embed.num_patches
        
        # Add class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        
        # Classifier head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize cls token
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize patch embedding parameters
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        
        # Initialize transformer blocks
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
    
    def forward(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        
        # Create patches and embed
        x = self.patch_embed(x)  # (B, N, E)
        
        # Add class token
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat((cls_token, x), dim=1)  # (B, N+1, E)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply classification head
        x = self.norm(x)
        if self.classifier == 'token':
            # Use class token
            x = x[:, 0]
        else:
            # Global average pooling
            x = x[:, 1:].mean(dim=1)
        
        x = self.head(x)
        return x
    
    def train_model(self, train_loader, val_loader, criterion, optimizer, num_epochs, device=None):
        """
        Train the VideoViT model
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.to(device)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train phase
            self.train()
            train_loss = 0.0
            train_corrects = 0
            
            for videos, labels in train_loader:
                videos = videos.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(videos)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                _, preds = torch.max(outputs, 1)
                train_loss += loss.item() * videos.size(0)
                train_corrects += torch.sum(preds == labels.data)
            
            epoch_train_loss = train_loss / len(train_loader.dataset)
            epoch_train_acc = train_corrects.double() / len(train_loader.dataset)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            val_corrects = 0
            
            with torch.no_grad():
                for videos, labels in val_loader:
                    videos = videos.to(device)
                    labels = labels.to(device)
                    
                    outputs = self(videos)
                    loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    val_loss += loss.item() * videos.size(0)
                    val_corrects += torch.sum(preds == labels.data)
            
            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_acc = val_corrects.double() / len(val_loader.dataset)
            
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')
            print(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')
            
            # Save best model
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                print(f'New best model with val loss: {best_val_loss:.4f}')
                torch.save(self.state_dict(), 'best_video_vit_model.pth')
        
        return self 