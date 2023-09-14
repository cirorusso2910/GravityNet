from pandas import read_csv
from sklearn.metrics import roc_curve
from typing import Tuple

from net.plot.coords.read_coords import read_coords
from net.metrics.utility.my_round_value import my_round_value


def FROC_FPS_index(detections_path: str,
                   coords_FROC_path: str,
                   FPS: int) -> Tuple[float, float]:
    """
    Get FROC sensitivity at specific FPS and score threshold of specific FPS

    :param detections_path: detections path
    :param coords_FROC_path: coords FROC path
    :param FPS: false positive per scan (FPS)
    :return: sensitivity at specific FPS,
             score threshold of specific FPS
    """

    # read detections test (numpy array)
    detections_test = read_csv(filepath_or_buffer=detections_path, usecols=["LABEL", "SCORE"]).dropna(subset='LABEL').values

    # read FROC coords
    FPS_test, sens_test = read_coords(coords_path=coords_FROC_path,
                                      coords_type='FROC')

    index = 0
    for n in range(len(FPS_test)):
        if FPS_test[n] >= FPS:
            index = n
            break

    # sensitivity FPS index
    sens_FPS = sens_test[index]

    label = detections_test[:, 0]  # label
    scores = detections_test[:, 1]  # scores

    # ROC curve
    FPR, TPR, thresholds = roc_curve(y_true=label, y_score=scores, pos_label=1)

    # score at FPS
    score_FPS = thresholds[index]

    print("\n---------------"
          "\nFROC FPS INDEX:"
          "\n---------------"
          "\nFPS : {}".format(FPS),
          "\nSensitivity: {}".format(my_round_value(sens_FPS, digits=3)),
          "\nScore: {}".format(my_round_value(score_FPS, digits=3)))

    return sens_FPS, score_FPS
