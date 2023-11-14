from typing import List


def recall(preds: List[str], targets: List[str]) -> float:
    """
    Calculate the recall score between predicted and target lists.

    Args:
        preds (List[str]): List of predicted values.
        targets (List[str]): List of target values.

    Returns:
        float: Recall score between 0.0 and 1.0.
    """

    if not targets:  # If targets list is empty
        if not preds:  # If preds list is also empty
            return 1.0  # Perfect recall since there are no targets to predict
        else:
            return 0.0  # No recall since there are no targets to predict

    true_positive = len(set(preds) & set(targets))
    recall_score = round(true_positive / len(targets), 4)

    return recall_score


def average_precision(preds: List[str], targets: List[str]) -> float:
    if not targets:
        if not preds:
            return 1.0
        else:
            return 0.0

    true_positive = [1 if pred in targets else 0 for pred in preds]

    cnt = 0
    average_precision_score = 0.0

    for i in range(len(true_positive)):
        if true_positive[i] == 1:
            cnt += 1
            average_precision_score += (cnt / (i + 1))

    if cnt == 0:
        return 0.0

    average_precision_score = round(average_precision_score / cnt, 4)
    return average_precision_score


def reciprocal_rank(preds: List[str], targets: List[str]) -> float:
    if not targets:
        if not preds:
            return 1.0
        else:
            return 0.0

    true_positive = [1 if pred in targets else 0 for pred in preds]

    for i in range(len(true_positive)):
        if true_positive[i] == 1:
            return round(1 / (i + 1), 4)

    return 0.0
