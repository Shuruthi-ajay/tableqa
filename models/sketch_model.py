import torch
from torch import nn
from transformers import AutoModel

OPS    = ["ADD", "SUB", "AVG", "COUNT", "NONE"]
SCALES = ["NONE", "THOUSAND", "MILLION", "PERCENT"]
TYPES  = ["NUMBER", "SPAN", "BOOLEAN"]

class SketchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("bert-large-uncased")
        h = self.encoder.config.hidden_size
        self.op = nn.Linear(h, len(OPS))
        self.scale = nn.Linear(h, len(SCALES))
        self.type = nn.Linear(h, len(TYPES))

    def forward(self, input_ids, attention_mask):
        cls = self.encoder(
            input_ids, attention_mask
        ).last_hidden_state[:, 0]

        return {
            "op": self.op(cls),
            "scale": self.scale(cls),
            "type": self.type(cls)
        }