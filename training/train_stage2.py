import torch
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from models.sketch_model import SketchModel
from data.load_tatqa import load_tatqa

# --------------------
# Device
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------
# Tokenizers (IMPORTANT: SEPARATE)
# --------------------
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
t5_tokenizer = AutoTokenizer.from_pretrained("t5-large")

# --------------------
# Latent sketch model (FROZEN)
# --------------------
sketch_model = SketchModel("bert-base-uncased").to(device)
sketch_model.eval()
for p in sketch_model.parameters():
    p.requires_grad = False

# --------------------
# Base model (T5)
# --------------------
MODEL_NAME = "t5-large"  # switch to t5-base if OOM
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

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

    for ex in train_data[:100]:  # small for Colab
        question = ex["question"]
        table = ex["table"]
        answer = str(ex["answer"])

        # ---- Sketch embedding (BERT tokenizer!)
        with torch.no_grad():
            enc_q = bert_tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(device)

            sketch_vec = sketch_model(**enc_q)

        # ---- Table → text
        table_text = " ".join(
            [" ".join(map(str, row)) for row in table[:5]]
        )

        input_text = (
            f"question: {question} "
            f"table: {table_text}"
        )

        enc = t5_tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=384
        ).to(device)

        labels = t5_tokenizer(
            answer,
            return_tensors="pt",
            truncation=True
        ).input_ids.to(device)

        outputs = model(**enc, labels=labels)
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
t5_tokenizer.save_pretrained("stage2_model")
print("Stage‑2 model saved")