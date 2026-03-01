import pandas as pd

def load_wikitq(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")
    data = []

    for _, r in df.iterrows():
        table = pd.read_csv(r["table_file"])
        data.append({
            "question": r["question"],
            "table": table,
            "answer": str(r["answer"]),
            "answer_type": "span",
            "scale": "NONE"
        })

    return data