import sys
from typing import List

from net.utility.msg.msg_error import msg_error


def metrics_header(metrics_type: str) -> List[str]:
    """
    Get metrics header according to type

    :param metrics_type: metrics type
    :return: header
    """

    if metrics_type == 'train':
        header = ["EPOCH",
                  "LOSS",
                  "CLASSIFICATION LOSS",
                  "REGRESSION LOSS",
                  "LEARNING RATE",
                  "AUC",
                  "SENSITIVITY WORK POINT",
                  "SENSITIVITY MAX",
                  "AUFROC [0, 1]",
                  "AUFROC [0, 10]",
                  "AUFROC [0, 50]",
                  "AUFROC [0, 100]",
                  "TIME TRAIN",
                  "TIME VALIDATION",
                  "TIME METRICS"]

    elif metrics_type == 'test':
        header = ["AUC",
                  "SENSITIVITY WORK POINT",
                  "SENSITIVITY MAX",
                  "AUFROC [0, 1]",
                  "AUFROC [0, 10]",
                  "AUFROC [0, 50]",
                  "AUFROC [0, 100]",
                  "TIME TEST",
                  "TIME METRICS"]

    elif metrics_type == 'test_NMS':
        header = ["AUC",
                  "SENSITIVITY WORK POINT",
                  "SENSITIVITY MAX",
                  "AUFROC [0, 1]",
                  "AUFROC [0, 10]",
                  "AUFROC [0, 50]",
                  "AUFROC [0, 100]",
                  "TIME NMS",
                  "TIME METRICS"]

    elif metrics_type == 'sensitivity':
        header = ["IMAGE",
                  "ANNOTATIONS",
                  "TP",
                  "FN",
                  "FP",
                  "SENSITIVITY"]

    else:
        str_err = msg_error(file=__file__,
                            variable=metrics_type,
                            type_variable='header metrics type',
                            choices='[train, test, test NMS, test complete, sensitivity]')
        sys.exit(str_err)

    return header
