import numpy as np
from sklearn.metrics import roc_auc_score


def AUC(detections: np.ndarray) -> float:
    """
    Compute Area Under the Curve (AUC)

    :param detections: detections
    :return: AUC value
    """

    label = detections[:, 0]  # label
    scores = detections[:, 1]  # scores

    # AUC
    try:
        AUC_value = roc_auc_score(y_true=label, y_score=scores, labels=1)
    except ValueError:
        AUC_value = 0

    return AUC_value
