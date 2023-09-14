import torch

from pandas import read_csv
from torch.utils.data import Dataset
from typing import Tuple

from net.initialization.header.statistics import statistics_header


def min_max_statistics(dataset: Dataset) -> Tuple[int, int]:
    """
    Compute (min, max) value of dataset

    :param dataset: dataset
    :return: (min, max) statistics
    """

    # init hist
    min_hist = []  # min history
    max_hist = []  # max history

    for i in range(dataset.__len__()):
        # sample
        sample = dataset[i]

        min_image = torch.min(sample['image'])  # min of original image
        max_image = torch.max(sample['image'])  # max of original image

        # append
        min_hist.append(min_image.item())
        max_hist.append(max_image.item())

    # min and max absolute
    min_abs = min(min_hist)
    max_abs = max(max_hist)

    return min_abs, max_abs


def read_min_max_statistics(statistics_path: str) -> dict:
    """
    Read (min, max) statistics

    :param statistics_path: statistics path
    :return: (min, max) statistics dictionary
    """

    header = statistics_header(statistics_type='min-max')
    statistics = read_csv(filepath_or_buffer=statistics_path, usecols=header)

    min_train = statistics['MIN'][0]
    max_train = statistics['MAX'][0]

    min_val = statistics['MIN'][1]
    max_val = statistics['MAX'][1]

    min_test = statistics['MIN'][2]
    max_test = statistics['MAX'][2]

    min_max_dict = {
        'train': {
            'min': min_train,
            'max': max_train,
        },

        'validation': {
            'min': min_val,
            'max': max_val,
        },

        'test': {
            'min': min_test,
            'max': max_test,
        },
    }

    return min_max_dict
