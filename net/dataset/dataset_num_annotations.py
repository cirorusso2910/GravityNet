import sys
import time

from torch.utils.data import Dataset

from net.dataset.statistics.num_annotations import get_num_annotations
from net.utility.msg.msg_error import msg_error


def dataset_num_annotations(split: str,
                            do_dataset_augmentation: bool,
                            dataset_train: Dataset,
                            dataset_val: Dataset,
                            dataset_test: Dataset) -> dict:
    """
    Compute dataset num annotations (calcifications)

    :param split: split name
    :param do_dataset_augmentation: do dataset augmentation
    :param dataset_train: dataset train
    :param dataset_val: dataset validation
    :param dataset_test: dataset test
    :return: dataset num annotations dictionary
    """

    # ------- #
    # DEFINED #
    # ------- #
    # split $N$-fold
    if split == '$N$-fold':
        num_annotations_train = '$NUM_ANNOTATIONS_TRAIN$'
        num_annotations_val = '$NUM_ANNOTATIONS_VALIDATION$'
        num_annotations_test = '$NUM_ANNOTATIONS_TEST$'

        if do_dataset_augmentation:
            num_annotations_train = '$NUM_ANNOTATIONS_TRAIN$' '*' '$NUM_AUGMENTATION_TRANSFORMS$'
            num_annotations_val = '$NUM_ANNOTATIONS_VALIDATION$'
            num_annotations_test = '$NUM_ANNOTATIONS_TEST$'

    # ---------- #
    # TO COMPUTE #
    # ---------- #
    else:
        time_annotations_start = time.time()

        # num annotations dataset-train
        num_annotations_train = get_num_annotations(dataset=dataset_train,
                                                    annotation_type_dict='annotation')

        # num annotations dataset-val
        num_annotations_val = get_num_annotations(dataset=dataset_val,
                                                  annotation_type_dict='annotation')

        # num annotations dataset-test
        num_annotations_test = get_num_annotations(dataset=dataset_test,
                                                   annotation_type_dict='annotation')

        time_annotations = time.time() - time_annotations_start

        print("Num Annotations computed in: {} m {} s".format(int(time_annotations) // 60, int(time_annotations) % 60))

    num_annotations = {
        'train': num_annotations_train,
        'validation': num_annotations_val,
        'test': num_annotations_test,
    }

    return num_annotations
