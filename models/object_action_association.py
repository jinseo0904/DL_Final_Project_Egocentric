"""
Implementation of the Object-Action Association Module using Cross-Attention.
"""
import torch
import torch.nn as nn

class ObjectActionAssociation(nn.Module):
    """
    Associates object features with action features using cross-attention.

    Assumes object features attend to a global action context vector.
    """
    def __init__(self, obj_dim: int, act_dim: int, embed_dim: int, num_heads: int):
        """
        Initializes the ObjectActionAssociation module.

        Args:
            obj_dim (int): Dimension of input object features.
            act_dim (int): Dimension of input action features.
            embed_dim (int): Embedding dimension for the attention mechanism.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.obj_dim = obj_dim
        self.act_dim = act_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Linear projections for Query, Key, Value
        self.query_proj = nn.Linear(obj_dim, embed_dim)
        self.key_proj = nn.Linear(act_dim, embed_dim)
        self.value_proj = nn.Linear(act_dim, embed_dim)

        # Multi-head attention layer
        # batch_first=False because MultiheadAttention expects (seq_len, batch_size, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

        # Optional: Layer normalization or further processing can be added here
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, object_features: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ObjectActionAssociation module.

        Args:
            object_features (torch.Tensor): Object features (batch_size, num_objects, obj_dim).
            action_features (torch.Tensor): Global action features (batch_size, act_dim).

        Returns:
            torch.Tensor: Fused object-action representations (batch_size, num_objects, embed_dim).
        """
        batch_size, num_objects, _ = object_features.shape

        # Project features into Q, K, V spaces
        # Query: Object features (batch_size, num_objects, obj_dim) -> (batch_size, num_objects, embed_dim)
        query = self.query_proj(object_features)

        # Key/Value: Action features (batch_size, act_dim) -> (batch_size, embed_dim)
        # We treat the single action feature vector as a sequence of length 1
        key = self.key_proj(action_features).unsqueeze(1) # (batch_size, 1, embed_dim)
        value = self.value_proj(action_features).unsqueeze(1) # (batch_size, 1, embed_dim)

        # Reshape for MultiheadAttention: (seq_len, batch_size, embed_dim)
        # Query: (num_objects, batch_size, embed_dim)
        query = query.permute(1, 0, 2)
        # Key/Value: (1, batch_size, embed_dim)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        # Apply cross-attention: Query attends to Key/Value
        # attn_output shape: (num_objects, batch_size, embed_dim)
        attn_output, attn_weights = self.attention(query=query, key=key, value=value)

        # Reshape back to (batch_size, num_objects, embed_dim)
        fused_features = attn_output.permute(1, 0, 2)

        # Apply layer normalization
        fused_features = self.norm(fused_features)

        # Optional: Add residual connection with object features if dimensions match
        # or add another linear layer to project back to obj_dim

        return fused_features

# Example usage (requires defining dimensions)
if __name__ == '__main__':
    # Define example dimensions
    BATCH_SIZE = 4
    NUM_OBJECTS = 10
    OBJ_DIM = 512  # Example dimension for YOLO features
    ACT_DIM = 768  # Example dimension for TimeSformer features (e.g., base model hidden size)
    EMBED_DIM = 256 # Internal dimension for attention
    NUM_HEADS = 8

    # Create dummy input tensors
    dummy_object_features = torch.randn(BATCH_SIZE, NUM_OBJECTS, OBJ_DIM)
    dummy_action_features = torch.randn(BATCH_SIZE, ACT_DIM)

    # Initialize the module
    association_module = ObjectActionAssociation(
        obj_dim=OBJ_DIM,
        act_dim=ACT_DIM,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS
    )

    # Perform forward pass
    fused_output = association_module(dummy_object_features, dummy_action_features)

    print(f"Input Object Features Shape: {dummy_object_features.shape}")
    print(f"Input Action Features Shape: {dummy_action_features.shape}")
    print(f"Fused Output Features Shape: {fused_output.shape}") # Should be (BATCH_SIZE, NUM_OBJECTS, EMBED_DIM) 