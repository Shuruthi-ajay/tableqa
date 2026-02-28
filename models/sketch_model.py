import torch
from torch import nn
from transformers import AutoModel

# --------------------
# Sketch label spaces
# --------------------
OPS = ["ADD", "SUB", "AVG", "COUNT", "NONE"]
SCALES = ["NONE", "THOUSAND", "MILLION", "PERCENT"]
TYPES = ["NUMBER", "SPAN", "BOOLEAN"]


class SketchModel(nn.Module):
    """
    Multi-head sketch predictor:
    - Operation
    - Scale
    - Answer Type
    """

    def __init__(self, encoder_name="bert-large-uncased"):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size

        self.op_head = nn.Linear(hidden, len(OPS))
        self.scale_head = nn.Linear(hidden, len(SCALES))
        self.type_head = nn.Linear(hidden, len(TYPES))

    def forward(self, input_ids, attention_mask, **kwargs):
        """
        **kwargs is IMPORTANT to safely ignore token_type_ids
        """

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS representation
        cls = outputs.last_hidden_state[:, 0]

        return {
            "op": self.op_head(cls),
            "scale": self.scale_head(cls),
            "type": self.type_head(cls)
        }