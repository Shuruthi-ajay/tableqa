import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from data.load_tatqa import load_tatqa
from utils.table import linearize_table

MODEL_NAME = "t5-base"
EPOCHS = 6
MAX_SAMPLES = 3000
LR = 3e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
optimizer = AdamW(model.parameters(), lr=LR)

train_data = load_tatqa(
    "../TAT-QA/dataset_raw/tatqa_dataset_train.json",
    split="train"
)

model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0

    for ex in train_data[:MAX_SAMPLES]:
        question = ex["question"]
        table = ex["table"]
        answer = str(ex["answer"])

        table_text = linearize_table(table)
        input_text = f"question: {question} table: {table_text}"

        enc = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=384
        ).to(device)

        labels = tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64
        ).input_ids.to(device)

        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(**enc, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Avg Loss: {total_loss / MAX_SAMPLES:.4f}")

model.save_pretrained("stage2_model")
tokenizer.save_pretrained("stage2_model")
print("Stage‑2 model saved")