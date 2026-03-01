import torch
import pandas as pd
from transformers import TapasTokenizer, TapasForSequenceClassification


class AnswerTypeTapas:
    """
    Predicts answer type using TAPAS:
    NUMBER vs TEXT
    """

    def __init__(self):
        self.tokenizer = TapasTokenizer.from_pretrained(
            "google/tapas-base-finetuned-wtq"
        )
        self.model = TapasForSequenceClassification.from_pretrained(
            "google/tapas-base-finetuned-wtq",
            num_labels=2
        )
        self.model.eval()

    def predict(self, question, table):
        df = pd.DataFrame(table)

        inputs = self.tokenizer(
            table=df,
            queries=[question],
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        label = torch.argmax(logits, dim=-1).item()
        return "NUMBER" if label == 1 else "TEXT"