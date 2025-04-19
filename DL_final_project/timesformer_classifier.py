from transformers import TimesformerModel, TimesformerConfig
import torch.nn as nn
import torch

class TimeSformerVideoClassifier(nn.Module):
    def __init__(self, num_classes=15, pretrained_model='facebook/timesformer-base-finetuned-k400', dropout=0.3, freeze_backbone=False):
        super().__init__()
        self.backbone = TimesformerModel.from_pretrained(pretrained_model)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_dim = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):  # x: [B, C, T, H, W]
        # TimeSformer expects: [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]  # CLS token
        logits = self.classifier(self.dropout(cls_token))
        return logits
