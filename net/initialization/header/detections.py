import sys
from typing import List

from net.utility.msg.msg_error import msg_error


def detections_header() -> List[str]:
    """
    Get detections header

    :return: header
    """

    # detections header
    header = ["FILENAME",
              "NUM PREDICTIONS",
              "LABEL",
              "SCORE",
              "PREDICTION X",
              "PREDICTION Y",
              "TARGET X",
              "TARGET Y"]

    return header
