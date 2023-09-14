from typing import Tuple

import numpy as np

from sklearn.metrics import roc_curve


def FROC(detections: np.ndarray,
         TotalNumOfImages: int,
         TotalNumOfAnnotations: int,
         debug: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FROC Curve

    :param detections: detections
    :param TotalNumOfImages: total num of images
    :param TotalNumOfAnnotations: total num of annotations
    :param debug: debug option
    :return: false positive per scan (FPS) and sensitivity
    """

    label = detections[:, 0]  # label
    scores = detections[:, 1]  # scores

    TotalNumOfAnchors = len(scores)
    NumOfDetected = sum(label)

    if debug:
        print("\nDEBUG: FROC (eval.py)",
              "\nTotalNumOfAnchors: ", TotalNumOfAnchors,
              "\nNumOfDetected: ", NumOfDetected,
              "\nTotalNumOfAnnotations: ", TotalNumOfAnnotations,
              "\nTotalNumOfImages: ", TotalNumOfImages)

    # ROC curve
    FPR, TPR, thresholds = roc_curve(y_true=label, y_score=scores, pos_label=1)

    # FROC curve
    FPS = FPR * (TotalNumOfAnchors - NumOfDetected) / TotalNumOfImages
    sens = (TPR * NumOfDetected) / TotalNumOfAnnotations

    return FPS, sens
