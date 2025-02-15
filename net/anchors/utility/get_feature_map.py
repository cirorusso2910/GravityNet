import numpy as np

from typing import Tuple


def get_feature_map(image_height: int,
                    image_width: int) -> Tuple[int, int]:
    """
    Get feature map shape

    :param image_height: image height shape
    :param image_width: image width shape
    :return: feature map
    """

    # image shape
    image_shape = np.array((image_height, image_width))  # H x W

    # compute the feature map shape
    p = 5  # level 5 (of FPN)
    strides = 2 ** p  # stride
    feature_map_shape = (image_shape + 2 ** p - 1) // (2 ** p)  # H_FM x W_FM

    return feature_map_shape
