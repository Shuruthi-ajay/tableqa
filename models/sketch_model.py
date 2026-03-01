import torch
from torch import nn
from transformers import AutoModel


class SketchModel(nn.Module):
    """
    Latent Sketch Encoder using T5 Encoder + Mean Pooling
    """

    def __init__(self, encoder_name="t5-base"):
        super().__init__()

        # Only the encoder part of T5
        self.encoder = AutoModel.from_pretrained(encoder_name).encoder
        self.hidden_size = self.encoder.config.d_model

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = outputs.last_hidden_state          # (B, L, H)
        mask = attention_mask.unsqueeze(-1)         # (B, L, 1)

        # Mean pooling
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        return pooled