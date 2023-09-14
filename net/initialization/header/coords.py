import sys

from typing import List


def coords_header(coords_type: str) -> List:
    """
    Get coords header according to type

    :param coords_type: coords type
    :return: header
    """

    if coords_type == 'FROC':
        header = ["FPS",
                  "SENSITIVITY"]

    elif coords_type == 'ROC':
        header = ["FPR",
                  "TPR"]

    else:
        str_err = "\nERROR in header.py" \
                  "\n{} wrong header coords type" \
                  "\n[FROC, ROC]".format(coords_type)
        sys.exit(str_err)

    return header
