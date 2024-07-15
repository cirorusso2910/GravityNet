from typing import Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve


def PR(detections: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall Curve

    :param detections: detections
    :return: precision and recall
    """

    label = detections[:, 0]  # label
    scores = detections[:, 1]  # scores

    # compute the Precision-Recall Curve (PR)
    precision, recall, thresholds = precision_recall_curve(label, scores)

    return precision, recall
