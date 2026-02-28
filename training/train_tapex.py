import torch
from torch.optim import AdamW
from transformers import TapexTokenizer
from models.conditioned_tapex import ConditionedTapex
from models.sketch_model import SketchModel
from data.load_tatqa import load_tatqa

# --------------------
# Setup
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = TapexTokenizer.from_pretrained(
    "microsoft/tapex-large-finetuned-wtq"
)

# --------------------
# Load pretrained Sketch (FROZEN)
# --------------------
sketch_model = SketchModel().to(device)
sketch_model.load_state_dict(torch.load("sketch_model.pt"))
sketch_model.eval()   # IMPORTANT: no rule-based learning, frozen latent sketch

# --------------------
# Conditioned TAPEX (Stage 2)
# --------------------
model = ConditionedTapex().to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

# --------------------
# Data
# --------------------
train_data = load_tatqa(
    "TAT-QA/dataset_raw/tatqa_dataset_train.json",
    split="train"
)

# --------------------
# Training
# --------------------
model.train()
EPOCHS = 3

for epoch in range(EPOCHS):
    total_loss = 0.0

    for ex in train_data:

        question = ex["question"]
        table = ex["table"]
        answer = str(ex["answer"])

        # ---- Sketch embedding (NO RULES, NO LABELS)
        with torch.no_grad():
            enc_sketch = tokenizer(
                question,
                return_tensors="pt"
            ).to(device)

            sketch_out = sketch_model(
                input_ids=enc_sketch["input_ids"],
                attention_mask=enc_sketch["attention_mask"]
            )

            # latent sketch vector (CLS-style)
            sketch_vector = (
                sketch_out["op"]
                + sketch_out["scale"]
                + sketch_out["type"]
            )

        # ---- TAPEX input
        enc = tokenizer(
            table=table,
            query=question,
            return_tensors="pt"
        )

        enc = {k: v.to(device) for k, v in enc.items()}

        labels = tokenizer(
            answer,
            return_tensors="pt"
        )["input_ids"].to(device)

        # ---- Stage‑2 forward (THIS IS WHAT YOU ASKED ABOUT)
        outputs = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            sketch_emb=sketch_vector,
            labels=labels
        )

        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1} | Loss: {total_loss:.4f}")

# --------------------
# Save
# --------------------
model.tapex.save_pretrained("trained_tapex")
tokenizer.save_pretrained("trained_tapex")

print("Stage‑2 TAPEX training complete.")