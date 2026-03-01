
# evaluation/protocol.py
def answer_question(*args, **kwargs):
    raise RuntimeError(
        "answer_question is deprecated. "
        "Use run_dev.py / run_test.py with direct T5 generation."
    )