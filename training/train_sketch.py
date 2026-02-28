import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.sketch_model import SketchModel, OPS, SCALES, TYPES
from data.load_tatqa import load_tatqa

# --------------------
# Setup
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

# --------------------
# Collate function
# --------------------
def collate_fn(batch):
    questions = [ex["question"] for ex in batch]

    enc = tokenizer(
        questions,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    labels = {
        "op": torch.tensor([OPS.index(ex["op"]) for ex in batch]),
        "scale": torch.tensor([SCALES.index(ex["scale"]) for ex in batch]),
        "type": torch.tensor([TYPES.index(ex["answer_type"]) for ex in batch]),
    }

    return enc, labels


# --------------------
# Load TAT-QA data
# --------------------
train_data = load_tatqa(
    "TAT-QA/dataset_raw/tatqa_dataset_train.json",
    split="train"
)

train_loader = DataLoader(
    train_data,
    batch_size=4,        # T4-safe
    shuffle=True,
    collate_fn=collate_fn
)

# --------------------
# Model + Optimizer
# --------------------
model = SketchModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# --------------------
# Training loop
# --------------------
model.train()

for epoch in range(3):
    total_loss = 0.0

    for enc, labels in train_loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        labels = {k: v.to(device) for k, v in labels.items()}

        outputs = model(**enc)

        loss_op = loss_fn(outputs["op"], labels["op"])
        loss_scale = loss_fn(outputs["scale"], labels["scale"])
        loss_type = loss_fn(outputs["type"], labels["type"])

        loss = loss_op + loss_scale + loss_type

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1} | Total Loss: {total_loss:.4f}")

# --------------------
# Save model
# --------------------
torch.save(model.state_dict(), "sketch_model.pt")
print("Saved sketch_model.pt")