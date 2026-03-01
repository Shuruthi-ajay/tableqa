import torch
from torch import nn
from transformers import AutoModel


class SketchModel(nn.Module):
    """
    Latent Sketch Encoder (Stage 2 only)

    - NO explicit operation / scale / type labels
    - NO rule-based supervision
    - Used as a frozen latent encoder
    - CLS embedding is treated as the latent sketch
    """

    def __init__(self, encoder_name: str = "bert-base-uncased"):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size

    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Accepts **kwargs to safely ignore token_type_ids
        """

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token embedding = latent sketch vector
        sketch_embedding = outputs.last_hidden_state[:, 0]

        return sketch_embedding