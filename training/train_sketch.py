import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.sketch_model import SketchModel, OPS, SCALES, TYPES
from data.load_tatqa import load_tatqa

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

def encode(batch):
    return tokenizer(
        batch["question"],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

train_data = load_tatqa(
    "TAT-QA/dataset_raw/tatqa_dataset_train.json", "train"
)

model = SketchModel().cuda()
optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    total_loss = 0
    for ex in train_data:
        enc = encode(ex)
        enc = {k: v.cuda() for k, v in enc.items()}

        out = model(**enc)

        op_loss = loss_fn(out["op"], torch.tensor([OPS.index(ex["op"])]).cuda())
        scale_loss = loss_fn(out["scale"], torch.tensor([SCALES.index(ex["scale"])]).cuda())
        type_loss = loss_fn(out["type"], torch.tensor([TYPES.index(ex["answer_type"])]).cuda())

        loss = op_loss + scale_loss + type_loss
        loss.backward()
        optim.step()
        optim.zero_grad()

        total_loss += loss.item()

    print(f"Epoch {epoch}: {total_loss:.2f}")

torch.save(model.state_dict(), "sketch_model.pt")