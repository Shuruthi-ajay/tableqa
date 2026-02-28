import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM


class ConditionedTapex(nn.Module):
    """
    TAPEX via generic Seq2Seq interface (Transformers v5+ compatible)
    """

    def __init__(self, sketch_dim=1024, num_sketch_tokens=3):
        super().__init__()

        self.tapex = AutoModelForSeq2SeqLM.from_pretrained(
            "microsoft/tapex-large-finetuned-wtq"
        )

        self.sketch_proj = nn.Linear(
            sketch_dim,
            self.tapex.config.d_model
        )

        self.num_sketch_tokens = num_sketch_tokens

    def forward(
        self,
        input_ids,
        attention_mask,
        sketch_emb,
        labels=None
    ):
        B = sketch_emb.size(0)

        # Project sketch → TAPEX hidden space
        sketch_tokens = self.sketch_proj(sketch_emb)
        sketch_tokens = sketch_tokens.unsqueeze(1)
        sketch_tokens = sketch_tokens.repeat(1, self.num_sketch_tokens, 1)

        # Original embeddings
        inputs_embeds = self.tapex.get_input_embeddings()(input_ids)

        # Prefix sketch tokens
        inputs_embeds = torch.cat(
            [sketch_tokens, inputs_embeds], dim=1
        )

        # Extend attention mask
        sketch_mask = torch.ones(
            B, self.num_sketch_tokens,
            device=input_ids.device
        )
        attention_mask = torch.cat(
            [sketch_mask, attention_mask], dim=1
        )

        return self.tapex(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )