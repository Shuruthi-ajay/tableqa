import torch
from torch.optim import AdamW
from models.dual_head_tapex import DualHeadTapex
from models.conditioned_tapex import tokenizer, encode_input
from models.evidence_masking import prune_table
from models.sketch_model import SketchModel
from data.load_tatqa import load_tatqa

device = "cuda"

sketch_model = SketchModel().to(device)
sketch_model.load_state_dict(torch.load("sketch_model.pt"))
sketch_model.eval()

base_model = DualHeadTapex.from_pretrained("microsoft/tapex-large")
base_model.to(device)

optimizer = AdamW(base_model.parameters(), lr=1e-5)

train_data = load_tatqa(
    "TAT-QA/dataset_raw/tatqa_dataset_train.json", "train"
)

LAMBDA = 0.3

for epoch in range(3):
    total_loss = 0

    for ex in train_data:
        with torch.no_grad():
            sketch = sketch_model.predict(ex["question"])

        table = prune_table(ex["table"], sketch)
        enc = encode_input(ex["question"], table, sketch)
        enc = {k: v.to(device) for k, v in enc.items()}

        labels = tokenizer(
            str(ex["answer"]),
            return_tensors="pt"
        )["input_ids"].to(device)

        gen_loss, sketch_logits = base_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels
        )

        sketch_loss = torch.mean(sketch_logits ** 2)
        loss = gen_loss + LAMBDA * sketch_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print(f"Epoch {epoch}: {total_loss:.2f}")

base_model.save_pretrained("trained_tapex")
tokenizer.save_pretrained("trained_tapex")