from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

CONTROL_TOKENS = [
    "<ANS_NUM>", "<ANS_SPAN>",
    "<OP_ADD>", "<OP_SUB>", "<OP_AVG>",
    "<SCALE_NONE>", "<SCALE_K>", "<SCALE_M>", "<SCALE_P>"
]

tokenizer = T5Tokenizer.from_pretrained("microsoft/tapex-large", legacy=False)
tokenizer.add_tokens(CONTROL_TOKENS)

model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/tapex-large")
model.resize_token_embeddings(len(tokenizer))

def encode_input(question, table, sketch):
    prefix = (
        f"<ANS_{sketch['type']}> "
        f"<OP_{sketch['op']}> "
        f"<SCALE_{sketch['scale']}> "
    )
    text = prefix + "Question: " + question + " Table: " + table.to_string(index=False)
    return tokenizer(text, return_tensors="pt", truncation=True)