import numpy as np
import torch


def init_detections_distance(num_predictions: int,
                             classification: torch.Tensor,
                             prediction: torch.Tensor,
                             device: torch.device) -> torch.Tensor:
    """
    initialize detections distance

    HEADER:
    - NUM PREDICTION: num prediction
    - LABEL: [1: TP | 0: FP | nan: FN]
    - SCORE: classification score
    - PREDICTION X: coord x gravity points prediction
    - PREDICTION Y: coord y gravity points prediction
    - TARGET X: coord x annotation detected
    - TARGET Y: coord y annotation detected

    :param num_predictions: num prediction
    :param classification: classification score
    :param prediction: predictions
    :param device: device
    :return: detections initialized
    """

    # init detections (tensor)
    detections = torch.zeros((num_predictions, 7), dtype=torch.float32).to(device)

    # init num predictions [NUM PREDICTIONS]
    detections[:, 0] = torch.arange(0, num_predictions)

    # init label [LABEL]
    detections[:, 1] = torch.zeros(num_predictions)

    # init scores of each prediction (col 1: foreground) [SCORE POSITIVE]
    detections[:, 2] = classification[:, 1]

    # init coord prediction y [PREDICTION X]
    detections[:, 3] = prediction[:, 0]

    # init coord prediction x [PREDICTION Y]
    detections[:, 4] = prediction[:, 1]

    # init coord annotation y [TARGET X]
    detections[:, 5] = torch.zeros(num_predictions) * np.nan

    # init coord annotation x [TARGET Y]
    detections[:, 6] = torch.zeros(num_predictions) * np.nan

    return detections
