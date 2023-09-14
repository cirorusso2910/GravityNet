from typing import List


def detections_header() -> List[str]:
    """
    Get detections header

    :return: header
    """

    header = ["FILENAME",
              "NUM PREDICTION",
              "LABEL",
              "SCORE",
              "PREDICTION X",
              "PREDICTION Y",
              "TARGET X",
              "TARGET Y",
              "RADIUS"
              ]

    return header
