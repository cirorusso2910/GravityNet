import sys

from torch.utils.data import Dataset

from net.dataset.statistics.num_normal_images import get_num_normal_images
from net.utility.msg.msg_error import msg_error


def dataset_num_normal_images(dataset: str,
                              split: str,
                              do_dataset_augmentation: bool,
                              dataset_train: Dataset,
                              dataset_val: Dataset,
                              dataset_test: Dataset) -> dict:
    """
    Compute dataset num normal images

    :param split: split name
    :param do_dataset_augmentation: do dataset augmentation
    :param dataset_train: dataset train
    :param dataset_val: dataset validation
    :param dataset_test: dataset test
    :return: dataset num normal images dictionary
    """

    # ------- #
    # DEFINED #
    # ------- #
    # E-ophtha-MA
    if dataset == 'E-ophtha-MA':

        # split 1-fold
        if split == '1-fold':
            num_normal_images_train = 94
            num_normal_images_val = 24
            num_normal_images_test = 115

            if do_dataset_augmentation:
                num_normal_images_train = 94 * 4
                num_normal_images_val = 24
                num_normal_images_test = 115

        # split 2-fold
        elif split == '2-fold':
            num_normal_images_train = 91
            num_normal_images_val = 24
            num_normal_images_test = 118

            if do_dataset_augmentation:
                num_normal_images_train = 91 * 4
                num_normal_images_val = 24
                num_normal_images_test = 118

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
            num_normal_images_train = 35
            num_normal_images_val = 19
            num_normal_images_test = 43

            # augmentation
            if do_dataset_augmentation:
                num_normal_images_train = 35 * 4
                num_normal_images_val = 19
                num_normal_images_test = 43

        # split 2-fold
        elif split == '2-fold':
            num_normal_images_train = 26
            num_normal_images_val = 17
            num_normal_images_test = 54

            # augmentation
            if do_dataset_augmentation:
                num_normal_images_train = 26 * 4
                num_normal_images_val = 17
                num_normal_images_test = 54

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

        # E-ophtha-MA
        if dataset == 'E-ophtha-MA':

            # num normal images in dataset-train
            num_normal_images_train = get_num_normal_images(dataset=dataset_train,
                                                            annotation_type_dict='annotation')

            # num normal images in dataset-validation
            num_normal_images_val = get_num_normal_images(dataset=dataset_val,
                                                          annotation_type_dict='annotation')

            # num normal images in dataset-test
            num_normal_images_test = get_num_normal_images(dataset=dataset_test,
                                                           annotation_type_dict='annotation')

        # INbreast
        elif dataset == 'INbreast':

            # num normal images in dataset-train
            num_normal_images_train = get_num_normal_images(dataset=dataset_train,
                                                            annotation_type_dict='annotation')

            # num normal images in dataset-validation
            num_normal_images_val = get_num_normal_images(dataset=dataset_val,
                                                          annotation_type_dict='annotation')

            # num normal images in dataset-test
            num_normal_images_test = get_num_normal_images(dataset=dataset_test,
                                                           annotation_type_dict='annotation')

        else:
            str_err = msg_error(file=__file__,
                                variable=dataset,
                                type_variable="dataset",
                                choices="[INbreast, E-ophtha-MA]")
            sys.exit(str_err)

    num_normal_images = {
        'train': num_normal_images_train,
        'validation': num_normal_images_val,
        'test': num_normal_images_test,
    }

    return num_normal_images
