def validate_test_gold_alignment(test_data, gold):
    test_uids = {ex["uid"] for ex in test_data}
    gold_uids = set(gold.keys())
    overlap = test_uids & gold_uids

    print("Test questions:", len(test_uids))
    print("Gold questions:", len(gold_uids))
    print("Evaluable overlap:", len(overlap))

    assert len(overlap) > 0, "No overlap between test and gold sets"

    return overlap