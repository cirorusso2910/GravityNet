import numpy as np
import pandas as pd

from pandas import read_csv

from net.initialization.header.detections import detections_header


def detections_concatenation(detections_path_1_fold,
                             detections_path_2_fold,
                             detections_concatenated_path):
    """
    Save detections 1-fold and 2-fold concatenation

    :param detections_path_1_fold: detections 1-fold path
    :param detections_path_2_fold:  detections 2-fold path
    :param detections_concatenated_path: detections concatenation path
    """

    # detections header
    header = detections_header()

    # read detections 1-fold
    detections_1_fold = read_csv(filepath_or_buffer=detections_path_1_fold, usecols=header).values
    print("detections 1-fold reading: COMPLETE")

    # read detections 2-fold
    detections_2_fold = read_csv(filepath_or_buffer=detections_path_2_fold, usecols=header).values
    print("detections 2-fold reading: COMPLETE")

    # detections complete
    detections_complete = np.concatenate((detections_1_fold, detections_2_fold), axis=0)
    detections_csv = pd.DataFrame(detections_complete)
    detections_csv.to_csv(detections_concatenated_path, mode='w', index=False, header=header, float_format='%g')
    print("detections complete saving: COMPLETE")
