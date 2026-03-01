import torch
import torch.nn as nn
from transformers import AutoModel

class SketchModel(nn.Module):
    def __init__(self, model_name="t5-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name).encoder

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # last_hidden_state: (batch, seq_len, hidden)
        hidden = outputs.last_hidden_state

        # Mean pooling (mask-aware)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        return pooled  # (batch, hidden_dim)