import numpy as np

from typing import Tuple


def get_feature_grid(image_height: int,
                     image_width: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get grid shifts on image in x and y

    :param image_height: image height shape
    :param image_width: image width shape
    :return: shifts on image
    """

    # image shape
    image_shape = np.array((image_height, image_width))  # H x W

    # compute the feature map shape
    p = 5  # level 5 (of FPN)
    strides = 2 ** p  # stride
    feature_map_shape = (image_shape + 2 ** p - 1) // (2 ** p)  # H_FM x W_FM

    shift_height = (np.arange(0, feature_map_shape[0])) * strides
    shift_width = (np.arange(0, feature_map_shape[1])) * strides

    return shift_height, shift_width
