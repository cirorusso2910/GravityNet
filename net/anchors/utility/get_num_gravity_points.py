from typing import Tuple

import numpy as np


def get_num_gravity_points(image_shape: Tuple[int, int],
                           step: int) -> dict:
    """
    Get num of gravity points per feature grid and per image

    :param image_shape: image shape
    :param step: step parameter
    :return: num gravity points per feature grid and per whole image
    """

    # image shape
    image_height, image_width = image_shape
    image_shape = np.array((image_height, image_width))  # H x W

    # feature map shape
    p = 5  # level 5 (of FPN) | num of pooling layer
    stride = 2 ** p  # stride
    feature_map_shape = (image_shape + 2 ** p - 1) // (2 ** p)  # H x W depends on image shape (rescale)
    image_height_FM, image_width_FM = feature_map_shape

    # feature grid shape
    feature_grid_shape = image_shape / feature_map_shape  # K x K
    K = feature_grid_shape[0]

    # num gravity points per feature grid
    num_gravity_points_per_feature_grid = pow((int((K - 2) / step) + 1), 2)

    # num gravity points per image
    num_gravity_points_per_image = num_gravity_points_per_feature_grid * image_height_FM * image_width_FM

    # num gravity points dict
    num_gravity_points = {
        'num_gravity_points_per_feature_grid': num_gravity_points_per_feature_grid,
        'num_gravity_points_per_image': num_gravity_points_per_image
    }

    return num_gravity_points
