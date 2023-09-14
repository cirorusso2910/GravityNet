import sys
import numpy as np
import torch

from typing import Tuple

from net.anchors.initial_config.initial_dice_config import initial_dice_config
from net.anchors.initial_config.initial_grid_config import initial_grid_config
from net.anchors.utility.shift import shift
from net.debug.debug_anchors import debug_anchors
from net.utility.msg.msg_config_complete import msg_config_complete
from net.utility.msg.msg_error import msg_error


def gravity_points_config(config: str,
                          image_shape: np.array,
                          device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Generate gravity points configuration:
    initial configuration is generated which is then shift on the feature map
    so that each pixel of the feature map has the initial configuration.

    :param config: configuration type
    :param image_shape: image shape
    :param device: device
    :return: gravity points configuration,
             initial gravity points configuration,
             feature map shape
    """

    # ----------------- #
    # FEATURE MAP SHAPE #
    # ----------------- #
    p = 5  # level 5 (of FPN)
    stride = 2 ** p  # stride
    feature_map_shape = (image_shape + 2 ** p - 1) // (2 ** p)  # H x W depends on image shape (rescale)

    # init all gravity points
    all_gravity_points = np.zeros((0, 2)).astype(np.float32)

    # ------------ #
    # CONFIG: GRID #
    # ------------ #
    if 'grid' in config:

        # step grid
        step = int(config.split('-')[1])

        # initial gravity points configuration grid
        gravity_points_initial_config = initial_grid_config(step=step,
                                                            image_shape=image_shape,
                                                            feature_map_shape=feature_map_shape)

    # ------------ #
    # CONFIG: DICE #
    # ------------ #
    elif 'dice' in config:

        # dice num [1, 2, 3, 4, 5]
        dice = int(config.split('-')[1])

        # check dice num
        if dice != 1 and dice != 2 and dice != 3 and dice != 4 and dice != 5:
            str_err = msg_error(file=__file__,
                                variable=dice,
                                type_variable="dice num",
                                choices="[1, 2, 3, 4, 5]")
            sys.exit(str_err)

        # initial gravity points configuration dice
        gravity_points_initial_config = initial_dice_config(dice=dice,
                                                            image_shape=image_shape,
                                                            feature_map_shape=feature_map_shape)

    else:
        str_err = msg_error(file=__file__,
                            variable=config,
                            type_variable="gravity points configuration",
                            choices="[grid, dice]")
        sys.exit(str_err)

    # ----- #
    # SHIFT #
    # ----- #
    # shift gravity points over whole image
    shifted_gravity_points = shift(feature_map_shape=feature_map_shape,
                                   stride=stride,
                                   gravity_initial_config=gravity_points_initial_config)

    # gravity point configuration (A x 2)
    gravity_points = np.append(all_gravity_points, shifted_gravity_points, axis=0)

    # convert to torch tensor
    gravity_points = torch.from_numpy(gravity_points)

    msg_config_complete(config_type=config)

    return gravity_points.to(device), gravity_points_initial_config, feature_map_shape
