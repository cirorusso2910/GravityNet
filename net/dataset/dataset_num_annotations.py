import sys
import time

from torch.utils.data import Dataset

from net.dataset.statistics.num_annotations import get_num_annotations
from net.utility.msg.msg_error import msg_error


def dataset_num_annotations(dataset: str,
                            split: str,
                            do_dataset_augmentation: bool,
                            dataset_train: Dataset,
                            dataset_val: Dataset,
                            dataset_test: Dataset) -> dict:
    """
    Compute dataset num annotations (calcifications)

    :param dataset: dataset name
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
    # E-ophtha-MA
    if dataset == 'E-ophtha-MA':

        # split 1-fold
        if split == '1-fold':
            num_annotations_train = 542
            num_annotations_val = 105
            num_annotations_test = 659

            if do_dataset_augmentation:
                num_annotations_train = 542 * 4
                num_annotations_val = 105
                num_annotations_test = 659

        # split 2-fold
        elif split == '2-fold':
            num_annotations_train = 552
            num_annotations_val = 107
            num_annotations_test = 647

            if do_dataset_augmentation:
                num_annotations_train = 552 * 4
                num_annotations_val = 107
                num_annotations_test = 647

        else:
            str_err = msg_error(file=__file__,
                                variable=split,
                                type_variable="E-ophtha-MA split",
                                choices="[1-fold, 2-fold]")
            sys.exit(str_err)

    # INbreast
    elif dataset == 'INbreast':

        # split 1-fold
        if split == '1-fold':
            num_annotations_train = 2408  # w48m14
            num_annotations_val = 516  # filtered calcifications (radius < 7)
            num_annotations_test = 2756  # filtered calcifications (radius < 7)

            if do_dataset_augmentation:
                num_annotations_train = 2408 * 4  # w48m14 (x4)
                num_annotations_val = 516  # filtered calcifications (radius < 7)
                num_annotations_test = 2756  # filtered calcifications (radius < 7)

        # split 2-fold
        elif split == '2-fold':
            num_annotations_train = 2051  # w48m14
            num_annotations_val = 724  # filtered calcifications (radius < 7)
            num_annotations_test = 2901  # filtered calcifications (radius < 7)

            if do_dataset_augmentation:
                num_annotations_train = 2051 * 4  # w48m14 (x4)
                num_annotations_val = 724  # filtered calcifications (radius < 7)
                num_annotations_test = 2901  # filtered calcifications (radius < 7)

        else:
            str_err = msg_error(file=__file__,
                                variable=split,
                                type_variable="INbreast split",
                                choices="[1-fold, 2-fold]")
            sys.exit(str_err)

    # ---------- #
    # TO COMPUTE #
    # ---------- #
    else:
        time_annotations_start = time.time()

        # E-ophtha-MA
        if dataset == 'E-ophtha-MA':

            # num annotations dataset-train
            num_annotations_train = get_num_annotations(dataset=dataset_train,
                                                        annotation_type_dict='annotation')

            # num annotations dataset-val
            num_annotations_val = get_num_annotations(dataset=dataset_val,
                                                      annotation_type_dict='annotation')

            # num annotations dataset-test
            num_annotations_test = get_num_annotations(dataset=dataset_test,
                                                       annotation_type_dict='annotation')

        # INbreast
        elif dataset == 'INbreast':

            # num annotations dataset-train
            num_annotations_train = get_num_annotations(dataset=dataset_train,
                                                        annotation_type_dict='annotation_w48m14')

            # num annotations dataset-val
            num_annotations_val = get_num_annotations(dataset=dataset_val,
                                                      annotation_type_dict='annotation')

            # num annotations dataset-test
            num_annotations_test = get_num_annotations(dataset=dataset_test,
                                                       annotation_type_dict='annotation')

        else:
            str_err = msg_error(file=__file__,
                                variable=dataset,
                                type_variable="dataset",
                                choices="[E-ophtha-MA, INbreast]")
            sys.exit(str_err)

        time_annotations = time.time() - time_annotations_start

        print("Num Annotations computed in: {} m {} s".format(int(time_annotations) // 60, int(time_annotations) % 60))

    num_annotations = {
        'train': num_annotations_train,
        'validation': num_annotations_val,
        'test': num_annotations_test,
    }

    return num_annotations
