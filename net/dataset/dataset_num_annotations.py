import sys
import time

from torch.utils.data import Dataset

from net.dataset.statistics.num_annotations import get_num_annotations
from net.dataset.statistics.read_statistics import read_statistics
from net.utility.msg.msg_error import msg_error


def dataset_num_annotations(statistics_path: str,
                            small_lesion: str) -> dict:
    """
    Compute dataset num annotations

    :param statistics_path: statistics path
    :param small_lesion: small lesion
    :return: dataset num annotations dictionary
    """

    # read statistics
    statistics = read_statistics(statistics_path=statistics_path,
                                 small_lesion=small_lesion)

    num_annotations = {
        'train': statistics['annotations']['train'],
        'validation': statistics['annotations']['validation'],
        'test': statistics['annotations']['test']
    }

    return num_annotations
