import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from data.load_tatqa import load_tatqa
from data.validate import validate_test_gold_alignment
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
# Load test data
# --------------------
test = load_tatqa(
    "../TAT-QA/dataset_raw/tatqa_dataset_test.json",
    split="test"
)

with open("../TAT-QA/dataset_raw/tatqa_dataset_test_gold.json") as f:
    gold_raw = json.load(f)

gold = {
    q["uid"]: normalize_answer(str(q["answer"]))
    for item in gold_raw
    for q in item["questions"]
}

uids = validate_test_gold_alignment(test, gold)

em = 0.0
f1 = 0.0

for ex in test:
    if ex["uid"] not in uids:
        continue

    question = ex["question"]
    table = ex["table"]
    gold_ans = gold[ex["uid"]]

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

    em += exact_match(pred, gold_ans)
    f1 += token_f1(pred, gold_ans)

print("TEST EM:", em / len(uids))
print("TEST F1:", f1 / len(uids))