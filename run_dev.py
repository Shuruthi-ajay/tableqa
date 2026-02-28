from data.load_tatqa import load_tatqa
from evaluation.metrics import exact_match, token_f1
from evaluation.protocol import answer_question

data = load_tatqa("TAT-QA/dataset_raw/tatqa_dataset_dev.json", "dev")

em = f1 = 0
for ex in data:
    sketch = {"op":"NONE","scale":"NONE","type":"SPAN"}
    pred = answer_question(ex, sketch)
    em += exact_match(pred, ex["answer"])
    f1 += token_f1(pred, ex["answer"])

print("DEV EM:", em/len(data))
print("DEV F1:", f1/len(data))