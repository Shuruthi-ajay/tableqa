import numpy as np
from models.evidence_masking import prune_table
from models.conditioned_tapex import encode_input, tokenizer, model

def answer_question(example, sketch):
    table = prune_table(example["table"], sketch)
    enc = encode_input(example["question"], table, sketch)

    out = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_length=32
    )

    return tokenizer.decode(out[0], skip_special_tokens=True)