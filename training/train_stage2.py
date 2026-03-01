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
# Single tokenizer (T5 only)
# --------------------
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# --------------------
# Sketch model (T5 encoder + pooling, frozen)
# --------------------
sketch_model = SketchModel("t5-base").to(device)
sketch_model.eval()
for p in sketch_model.parameters():
    p.requires_grad = False

# --------------------
# Generator model (T5)
# --------------------
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

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
EPOCHS = 2

for epoch in range(EPOCHS):
    total_loss = 0.0

    for ex in train_data[:100]:  # small subset for Colab
        question = ex["question"]
        table = ex["table"]
        answer = str(ex["answer"])

        # ---- Sketch encoding (T5 encoder)
        with torch.no_grad():
            enc_q = tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
                max_length=128
            ).to(device)

            sketch_vec = sketch_model(
                input_ids=enc_q["input_ids"],
                attention_mask=enc_q["attention_mask"]
            )

        # ---- Table linearization
        table_text = " ; ".join(
            [" | ".join(map(str, row)) for row in table[:5]]
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
            max_length=64
        ).input_ids.to(device)

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

    print(f"Epoch {epoch + 1} | Loss: {total_loss:.3f}")

# --------------------
# Save model
# --------------------
model.save_pretrained("stage2_model")
tokenizer.save_pretrained("stage2_model")
print("Stage‑2 model saved")