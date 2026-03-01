import json
import pandas as pd

def load_tatqa(path, split):
    """
    split ∈ {"train", "dev", "test"}
    """
    with open(path) as f:
        data = json.load(f)

    examples = []

    for item in data:
        raw = item["table"]["table"]
        header, rows = raw[0], raw[1:]
        table = pd.DataFrame(rows, columns=header)

        for q in item["questions"]:
            ex = {
                "uid": q["uid"],
                "question": q["question"],
                "table": table
            }

            if split != "test":
                ex["answer"] = str(q["answer"])
                ex["answer_type"] = q["answer_type"]
                ex["scale"] = q.get("scale", "NONE")

            examples.append(ex)

    return examples