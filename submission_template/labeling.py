"""Optional. Decide whether to reveal the current item's label (True) or skip permanently (False).
Without this file the platform picks labels at random.
context keys: subject_id, benchmark_id, labels_acquired, labels_remaining, max_labels (31), items_remaining
"""


def acquisition_function(input, prediction, labeled, context):
    # uncertainty sampling: reveal when prediction is near 0.5, or when forced by remaining budget
    if context["labels_remaining"] >= context["items_remaining"]:
        return True
    return abs(prediction - 0.5) < 0.15
