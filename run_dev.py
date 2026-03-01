import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from data.load_tatqa import load_tatqa
from evaluation.metrics import exact_match, token_f1


def normalize_answer(ans):
    ans = ans.lower().strip()
    ans = ans.replace(",", "").replace("$", "").replace("%", "")
    ans = re.sub(r"\s+", " ", ans)
    return ans


# --------------------
# Device
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Load trained model
# --------------------
tokenizer = AutoTokenizer.from_pretrained("stage2_model")
model = AutoModelForSeq2SeqLM.from_pretrained("stage2_model").to(device)
model.eval()

# --------------------
# Load dev data
# --------------------
data = load_tatqa(
    "TAT-QA/dataset_raw/tatqa_dataset_dev.json",
    split="dev"
)

em = 0.0
f1 = 0.0

for ex in data:
    question = ex["question"]
    table = ex["table"]
    gold = normalize_answer(str(ex["answer"]))

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

    with torch.no_grad():
        output_ids = model.generate(
            **enc,
            max_length=64,
            num_beams=4
        )

    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    pred = normalize_answer(pred)

    em += exact_match(pred, gold)
    f1 += token_f1(pred, gold)

print("DEV EM:", em / len(data))
print("DEV F1:", f1 / len(data))