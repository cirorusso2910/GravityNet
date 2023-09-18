from typing import List


def detections_distance_header() -> List[str]:
    """
    Get detections distance header

    :return: header
    """

    header = ["FILENAME",
              "NUM PREDICTIONS",
              "LABEL",
              "SCORE",
              "PREDICTION X",
              "PREDICTION Y",
              "TARGET X",
              "TARGET Y"]

    return header


def detections_radius_header() -> List[str]:
    """
    Get detections radius header

    :return: header
    """

    header = ["FILENAME",
              "NUM PREDICTIONS",
              "LABEL",
              "SCORE",
              "PREDICTION X",
              "PREDICTION Y",
              "TARGET X",
              "TARGET Y",
              "RADIUS"]

    return header