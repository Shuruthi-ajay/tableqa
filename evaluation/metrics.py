def exact_match(pred, gold):
    return int(pred.strip() == gold.strip())

def token_f1(pred, gold):
    p = pred.split()
    g = gold.split()
    common = set(p) & set(g)
    if not common:
        return 0.0
    precision = len(common) / len(p)
    recall = len(common) / len(g)
    return 2 * precision * recall / (precision + recall)