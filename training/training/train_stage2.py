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
# Load latent sketch model (FROZEN)
# --------------------
sketch_model = SketchModel().to(device)
sketch_model.load_state_dict(
    torch.load("sketch_model.pt", map_location=device)
)
sketch_model.eval()

# --------------------
# Base model (T5 / BART instead of TAPEX)
# --------------------
MODEL_NAME = "t5-large"   # you can switch to "facebook/bart-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)

# --------------------
# Load TAT-QA data
# --------------------
train_data = load_tatqa(
    "TAT-QA/dataset_raw/tatqa_dataset_train.json",
    split="train"
)

# --------------------
# Training
# --------------------
model.train()
EPOCHS = 2

for epoch in range(EPOCHS):
    total_loss = 0.0

    for ex in train_data[:100]:   # small subset for Colab
        question = ex["question"]
        table = ex["table"]
        answer = str(ex["answer"])

        # ---- Latent sketch embedding (NO ops, NO rules)
        with torch.no_grad():
            enc_q = tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(device)

            sketch_vec = sketch_model(**enc_q)  # (1, hidden)

        # ---- Convert table → text (simple linearization)
        table_text = " ".join(
            [" ".join(map(str, row)) for row in table[:5]]
        )

        input_text = (
            f"question: {question} "
            f"table: {table_text}"
        )

        enc = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        labels = tokenizer(
            answer,
            return_tensors="pt",
            truncation=True
        ).input_ids.to(device)

        # ---- Forward + loss
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
tokenizer.save_pretrained("stage2_model")
print("Stage‑2 model saved")