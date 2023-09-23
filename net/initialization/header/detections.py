import sys
from typing import List

from net.utility.msg.msg_error import msg_error


def detections_header(eval: str) -> List[str]:
    """
    Get detections header

    :param eval: evaluation type
    :return: header
    """

    # detections distance header
    if 'distance' in eval:
        header = ["FILENAME",
                  "NUM PREDICTIONS",
                  "LABEL",
                  "SCORE",
                  "PREDICTION X",
                  "PREDICTION Y",
                  "TARGET X",
                  "TARGET Y"]

    # detections radius header
    elif 'radius' in eval:
        header = ["FILENAME",
                  "NUM PREDICTIONS",
                  "LABEL",
                  "SCORE",
                  "PREDICTION X",
                  "PREDICTION Y",
                  "TARGET X",
                  "TARGET Y",
                  "RADIUS"]

    else:
        str_err = msg_error(file=__file__,
                            variable=eval,
                            type_variable="evaluation",
                            choices="[distance, radius]")
        sys.exit(str_err)


    return header
