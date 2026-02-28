import json
from data.load_tatqa import load_tatqa
from data.validate import validate_test_gold_alignment
from evaluation.metrics import exact_match, token_f1
from evaluation.protocol import answer_question

test = load_tatqa("TAT-QA/dataset_raw/tatqa_dataset_test.json", "test")

with open("TAT-QA/dataset_raw/tatqa_dataset_test_gold.json") as f:
    gold_raw = json.load(f)

gold = {
    q["uid"]: str(q["answer"])
    for item in gold_raw
    for q in item["questions"]
}

uids = validate_test_gold_alignment(test, gold)

em = f1 = 0
for ex in test:
    if ex["uid"] not in uids:
        continue
    sketch = {"op":"NONE","scale":"NONE","type":"SPAN"}
    pred = answer_question(ex, sketch)
    em += exact_match(pred, gold[ex["uid"]])
    f1 += token_f1(pred, gold[ex["uid"]])

print("TEST EM:", em/len(uids))
print("TEST F1:", f1/len(uids))