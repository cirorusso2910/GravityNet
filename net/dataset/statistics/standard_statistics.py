import torch

from pandas import read_csv
from torch.utils.data import Dataset
from typing import Tuple

from net.initialization.header.statistics import statistics_header


def standard_statistics(dataset: Dataset) -> Tuple[float, float]:
    """
    Compute (mean, std) value of dataset

    :param dataset: dataset
    :return: (mean, std) statistics
    """

    # init hist
    mean_hist = []  # mean history
    mean_squared_hist = []  # mean squared history

    for i in range(dataset.__len__()):
        # sample
        sample = dataset[i]

        mean_image = torch.mean(sample['image'])  # mean of original image
        mean_squared = torch.mean(sample['image'] ** 2)  # mean squared of original image

        # append mean and std
        mean_hist.append(mean_image.item())
        mean_squared_hist.append(mean_squared.item())

    # mean
    mean_abs = sum(mean_hist) / dataset.__len__()

    # std = sqrt(E[X^2] - (E[X])^2)
    std_abs = (sum(mean_squared_hist) / dataset.__len__() - mean_abs ** 2) ** 0.5

    return mean_abs, std_abs


def read_std_statistics(statistics_path: str) -> dict:
    """
    Read (mean, std) statistics

    :param statistics_path: statistics path
    :return: (mean, std) statistics dictionary
    """

    header = statistics_header(statistics_type='std',
                               small_lesion_type='')
    statistics = read_csv(filepath_or_buffer=statistics_path, usecols=header)

    mean_train = statistics['MEAN'][0]
    std_train = statistics['STD'][0]

    mean_val = statistics['MEAN'][1]
    std_val = statistics['STD'][1]

    mean_test = statistics['MEAN'][2]
    std_test = statistics['STD'][2]

    std_dict = {
        'train': {
            'mean': mean_train,
            'std': std_train,
        },

        'validation': {
            'mean': mean_val,
            'std': std_val,
        },

        'test': {
            'mean': mean_test,
            'std': std_test,
        },
    }

    return std_dict
