import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM

class DualHeadTapex(nn.Module):
    def __init__(self, base_model, sketch_dim):
        super().__init__()
        self.model = base_model
        h = base_model.config.d_model
        self.sketch_head = nn.Linear(h, sketch_dim)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )

        enc_cls = out.encoder_last_hidden_state[:, 0]
        sketch_logits = self.sketch_head(enc_cls)

        return out.loss, sketch_logits