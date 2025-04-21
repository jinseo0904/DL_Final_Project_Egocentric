"""
Implementation of the Behavioral Context Inference Module using a Transformer Encoder.
"""
import torch
import torch.nn as nn

class BehavioralContextTransformer(nn.Module):
    """
    Infers high-level behavioral context from fused object-action representations
    using a Transformer Encoder.
    """
    def __init__(self, input_dim: int, num_classes: int, num_layers: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        """
        Initializes the BehavioralContextTransformer module.

        Args:
            input_dim (int): Dimension of the input fused features (output dim of association module).
            num_classes (int): Number of behavioral context classes to predict.
            num_layers (int): Number of sub-encoder-layers in the encoder.
            num_heads (int): Number of heads in the multiheadattention models.
            dim_feedforward (int): Dimension of the feedforward network model.
            dropout (float): Dropout value.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Expect input as (batch, seq, feature)
        )

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the BehavioralContextTransformer.

        Args:
            fused_features (torch.Tensor): Fused object-action representations
                                         (batch_size, num_objects, input_dim).

        Returns:
            torch.Tensor: Logits for behavioral context classes (batch_size, num_classes).
        """
        # Pass through transformer encoder
        # Input shape: (batch_size, num_objects, input_dim)
        # Output shape: (batch_size, num_objects, input_dim)
        encoded_features = self.transformer_encoder(fused_features)

        # Aggregate features across the 'num_objects' dimension (sequence length)
        # Mean pooling is a simple way to do this
        # Output shape: (batch_size, input_dim)
        aggregated_features = encoded_features.mean(dim=1)

        # Pass through classification head
        # Output shape: (batch_size, num_classes)
        logits = self.fc(aggregated_features)

        return logits

# Example usage
if __name__ == '__main__':
    # Define example dimensions consistent with the association module output
    BATCH_SIZE = 4
    NUM_OBJECTS = 10
    INPUT_DIM = 256 # Must match EMBED_DIM from ObjectActionAssociation
    NUM_CLASSES = 15 # As defined in strategy.md

    # Transformer parameters
    NUM_LAYERS = 3
    NUM_HEADS = 8
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1

    # Create dummy input tensor (output from the association module)
    dummy_fused_features = torch.randn(BATCH_SIZE, NUM_OBJECTS, INPUT_DIM)

    # Initialize the module
    behavior_module = BehavioralContextTransformer(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    )

    # Perform forward pass
    output_logits = behavior_module(dummy_fused_features)

    print(f"Input Fused Features Shape: {dummy_fused_features.shape}")
    print(f"Output Logits Shape: {output_logits.shape}") # Should be (BATCH_SIZE, NUM_CLASSES) 