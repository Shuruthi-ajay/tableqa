import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from models.sketch_model import SketchModel
from data.load_tatqa import load_tatqa

# --------------------
# Device
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------
# Tokenizer (T5 only)
# --------------------
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# --------------------
# Sketch model (NOT injected, frozen)
# --------------------
sketch_model = SketchModel("t5-base").to(device)
sketch_model.eval()
for p in sketch_model.parameters():
    p.requires_grad = False

# --------------------
# Generator model
# --------------------
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to(device)
optimizer = AdamW(model.parameters(), lr=5e-6)

# --------------------
# Load data
# --------------------
train_data = load_tatqa(
    "../TAT-QA/dataset_raw/tatqa_dataset_train.json",
    split="train"
)

# --------------------
# Training
# --------------------
model.train()
EPOCHS = 5
MAX_SAMPLES = 2000  # increase data

for epoch in range(EPOCHS):
    total_loss = 0.0

    for ex in train_data[:MAX_SAMPLES]:
        question = ex["question"]
        table = ex["table"]
        answer = str(ex["answer"])

        # ---- Optional sketch encoding (NOT used)
        with torch.no_grad():
            enc_q = tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
                max_length=128
            ).to(device)

            _ = sketch_model(
                input_ids=enc_q["input_ids"],
                attention_mask=enc_q["attention_mask"]
            )

        # ---- Better table linearization
        table_text = " ; ".join(
            [f"row{i}: " + " | ".join(map(str, row))
             for i, row in enumerate(table[:5])]
        )

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
            truncation=True,
            padding="max_length",
            max_length=64
        ).input_ids.to(device)

        # mask padding
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels
        )

        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / min(len(train_data), MAX_SAMPLES)
    print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

# --------------------
# Save model
# --------------------
model.save_pretrained("stage2_model")
tokenizer.save_pretrained("stage2_model")
print("Stage‑2 model saved")