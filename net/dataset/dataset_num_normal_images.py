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
    # $DATASET$
    if dataset == '$DATASET$':

        # split $N$-fold
        if split == '$N$-fold':
            num_normal_images_train = '$NUM_NORMAL_IMAGES_TRAIN$'
            num_normal_images_val = '$NUM_NORMAL_IMAGES_VALIDATION$'
            num_normal_images_test = '$NUM_NORMAL_IMAGES_TEST$'

            if do_dataset_augmentation:
                num_normal_images_train = '$NUM_NORMAL_IMAGES_TRAIN$' '*' '$NUM_AUGMENTATION_TRANSFORMS$'
                num_normal_images_val = '$NUM_NORMAL_IMAGES_VALIDATION$'
                num_normal_images_test = '$NUM_NORMAL_IMAGES_TEST$'

        else:
            str_err = msg_error(file=__file__,
                                variable=split,
                                type_variable="$DATASET$ split",
                                choices="[$N$-fold]")
            sys.exit(str_err)

    # ---------- #
    # TO COMPUTE #
    # ---------- #
    else:

        # $DATASET$
        if dataset == '$DATASET$':

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
                                choices="[$DATASET$]")
            sys.exit(str_err)

    num_normal_images = {
        'train': num_normal_images_train,
        'validation': num_normal_images_val,
        'test': num_normal_images_test,
    }

    return num_normal_images
