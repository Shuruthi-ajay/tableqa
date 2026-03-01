import re
from collections import Counter


def normalize_text(s):
    """
    Lowercase, remove punctuation/articles, normalize whitespace.
    """
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def exact_match(prediction, ground_truth):
    """
    Exact string match after normalization.
    """
    return int(normalize_text(prediction) == normalize_text(ground_truth))


def token_f1(prediction, ground_truth):
    """
    Token-level F1 score.
    """
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return (2 * precision * recall) / (precision + recall)