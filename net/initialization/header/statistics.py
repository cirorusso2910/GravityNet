import sys

from typing import List

from net.utility.msg.msg_error import msg_error


def statistics_header(statistics_type: str,
                      small_lesion_type: str) -> List:
    """
    Get statistics header

    :param statistics_type: statistics type
    :return: header
    """

    if statistics_type == 'statistics':
        header = ["DATASET",
                  "IMAGES",
                  "NORMALS",
                  "{}".format(small_lesion_type.upper()),
                  "MIN",
                  "MAX",
                  "MEAN",
                  "STD"
                  ]

    elif statistics_type == 'min-max':
        header = ["DATASET",
                  "MIN",
                  "MAX"]

    elif statistics_type == 'std':
        header = ["DATASET",
                  "MEAN",
                  "STD"]

    else:
        str_err = msg_error(file=__file__,
                            variable=statistics_type,
                            type_variable="statistics type",
                            choices="[statistics, min-max, std]")
        sys.exit(str_err)

    return header
