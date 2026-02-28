import torch
from torch import nn
from transformers import TapexForConditionalGeneration

class ConditionedTapex(nn.Module):
    """
    TAPEX conditioned via learned sketch prefix embeddings
    """

    def __init__(self, sketch_dim=768, num_sketch_tokens=3):
        super().__init__()

        self.tapex = TapexForConditionalGeneration.from_pretrained(
            "microsoft/tapex-large-finetuned-wtq"
        )

        self.sketch_proj = nn.Linear(sketch_dim, 
                                     self.tapex.config.d_model)

        self.num_sketch_tokens = num_sketch_tokens

    def forward(
        self,
        input_ids,
        attention_mask,
        sketch_emb,
        labels=None
    ):
        """
        sketch_emb: (B, sketch_dim)
        """

        B = sketch_emb.size(0)

        # project sketch → TAPEX space
        sketch_tokens = self.sketch_proj(sketch_emb)
        sketch_tokens = sketch_tokens.unsqueeze(1)
        sketch_tokens = sketch_tokens.repeat(1, self.num_sketch_tokens, 1)

        # prepend sketch tokens
        inputs_embeds = self.tapex.model.encoder.embed_tokens(input_ids)
        inputs_embeds = torch.cat([sketch_tokens, inputs_embeds], dim=1)

        # extend attention mask
        sketch_mask = torch.ones(
            B, self.num_sketch_tokens, device=input_ids.device
        )
        attention_mask = torch.cat([sketch_mask, attention_mask], dim=1)

        return self.tapex(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )