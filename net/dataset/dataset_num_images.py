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
    # E-ophtha-MA
    if dataset == 'E-ophtha-MA':

        # split 1-fold
        if split == '1-fold':
            num_images_train = 154
            num_images_val = 38
            num_images_test = 189

            if do_dataset_augmentation:
                num_images_train = 154 * 4
                num_images_val = 38
                num_images_test = 189

        # split 2-fold
        elif split == '2-fold':
            num_images_train = 151
            num_images_val = 38
            num_images_test = 192

            if do_dataset_augmentation:
                num_images_train = 151 * 4
                num_images_val = 38
                num_images_test = 192

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
            num_images_train = 143
            num_images_val = 62
            num_images_test = 205

            # augmentation
            if do_dataset_augmentation:
                num_images_train = 143 * 4
                num_images_val = 62
                num_images_test = 205

        # split 2-fold
        elif split == '2-fold':
            num_images_train = 143
            num_images_val = 62
            num_images_test = 205

            # augmentation
            if do_dataset_augmentation:
                num_images_train = 143 * 4
                num_images_val = 62
                num_images_test = 205

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
        num_images_train = dataset_train.__len__()
        num_images_val = dataset_val.__len__()
        num_images_test = dataset_test.__len__()

    num_images = {
        'train': num_images_train,
        'validation': num_images_val,
        'test': num_images_test
    }

    return num_images
