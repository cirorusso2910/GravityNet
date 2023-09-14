from typing import Tuple

import numpy as np
from sklearn.metrics import roc_curve


def ROC(detections: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ROC Curve

    :param detections: detections
    :return: False Positive Rate (FPR) and True Positive Rate (TPR)
    """
    """ ROC Curve """

    label = detections[:, 0]  # label
    scores = detections[:, 1]  # scores

    # ROC curve
    FPR, TPR, thresholds = roc_curve(y_true=label, y_score=scores, pos_label=1)

    return FPR, TPR
