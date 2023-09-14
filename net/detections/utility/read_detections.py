import sys

import numpy as np
from pandas import read_csv

from net.utility.msg.msg_error import msg_error


def read_detections(type_detections: str,
                    path_dict: dict) -> np.ndarray:
    """
    Read detections file (.csv) according to type

    :param type_detections: type detections file
    :param path_dict: path dictionary
    :return: detections
    """

    if type_detections == 'test':
        detections = read_csv(filepath_or_buffer=path_dict['detections']['test'], usecols=["FILENAME", "LABEL", "SCORE"]).values
    elif type_detections == 'test-NMS={}'.format(type_detections.split('=')[1]):
        detections = read_csv(filepath_or_buffer=path_dict['detections']['test_NMS'][type_detections.split('=')[1]], usecols=["FILENAME", "LABEL", "SCORE"]).values
    else:
        str_err = msg_error(file=__file__,
                            variable=type_detections,
                            type_variable='type detections',
                            choices='[test, test-NMS]')
        sys.exit(str_err)

    return detections
