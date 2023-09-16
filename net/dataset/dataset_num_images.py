import sys

from torch.utils.data import Dataset

from net.utility.msg.msg_error import msg_error


def dataset_num_images(dataset: str,
                       split: str,
                       do_dataset_augmentation: bool,
                       dataset_train: Dataset,
                       dataset_val: Dataset,
                       dataset_test: Dataset) -> dict:
    """
    Compute dataset num images

    :param dataset: dataset name
    :param split: split name
    :param do_dataset_augmentation: do dataset augmentation
    :param dataset_train: dataset train
    :param dataset_val: dataset validation
    :param dataset_test: dataset test
    :return: dataset num images dictionary
    """

    # ------- #
    # DEFINED #
    # ------- #
    # $DATASET$
    if dataset == '$DATASET$':

        # split $N$-fold
        if split == '$N$-fold':
            num_images_train = '$NUM_IMAGES_TRAIN$'
            num_images_val = '$NUM_IMAGES_VALIDATION$'
            num_images_test = '$NUM_IMAGES_TEST$'

            if do_dataset_augmentation:
                num_images_train = '$NUM_IMAGES_TRAIN$' '*' '$NUM_AUGMENTATION_TRANSFORMS$'
                num_images_val = '$NUM_IMAGES_VALIDATION$'
                num_images_test = '$NUM_IMAGES_TEST$'

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
        num_images_train = dataset_train.__len__()

        if do_dataset_augmentation:
            num_images_train = dataset_train.__len__() * '$NUM_AUGMENTATION_TRANSFORMS$'

        num_images_val = dataset_val.__len__()
        num_images_test = dataset_test.__len__()

    num_images = {
        'train': num_images_train,
        'validation': num_images_val,
        'test': num_images_test
    }

    return num_images
