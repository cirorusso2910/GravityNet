import sys

from net.utility.msg.msg_error import msg_error


def select_FP_list_path(FP_images: str,
                        path: dict) -> str:
    """
    Select list of images which calculate False Positive (FP)

    :param FP_images: images where calculate FP
    :param path: path dictionary
    :return: path images list
    """

    # FP calculated on all images
    if FP_images == 'all':
        FP_list_path = path['lists']['all']

    # FP calculated only on images normals (with no lesions)
    elif FP_images == 'normals':
        FP_list_path = path['lists']['normals']

    else:
        str_err = msg_error(file=__file__,
                            variable=FP_images,
                            type_variable="FP images",
                            choices="[all, normals]")
        sys.exit(str_err)

    return FP_list_path
