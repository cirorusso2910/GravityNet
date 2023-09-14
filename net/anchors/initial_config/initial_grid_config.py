import math
import numpy as np
import torch


def initial_grid_config(step: int,
                        image_shape: np.ndarray,
                        feature_map_shape: np.ndarray) -> torch.Tensor:
    """
    Generate initial gravity points with grid configuration in a reference window (feature map)

    :param step: step
    :param image_shape: image shape
    :param feature_map_shape: feature map shape
    :return: initial gravity points with grid configuration
    """

    # feature map grid on image (H x W)
    grid_height = math.ceil(image_shape[0] / feature_map_shape[0])
    grid_width = math.ceil(image_shape[1] / feature_map_shape[1])

    # initial configuration
    init_config_gravity_points = np.zeros((0, 2)).astype(np.int)

    # init gap counter for grid
    gap_counter_cols = 0  # gap counter cols
    gap_counter_rows = 0  # gap counter rows

    # ---- #
    # GRID #
    # ---- #
    # in feature grid are generated a grid configuration using two 'for' loop in x and y
    for y in range(int(-grid_height * 0.5), int(grid_height * 0.5)):
        for x in range(int(-grid_width * 0.5), int(grid_width * 0.5)):

            # anchor generation order
            #   10 -> 11 ->  12 -> ...
            #   7  -> 8  ->  9
            #   4  -> 5  ->  6
            #   1  -> 2  ->  3

            # condition to "set" gravity points
            if gap_counter_rows == 0 and gap_counter_cols == 0:
                coord = [x, y]
                init_config_gravity_points = np.append(init_config_gravity_points, [coord], axis=0)

            gap_counter_cols += 1
            if gap_counter_cols == step:
                gap_counter_cols = 0

        gap_counter_rows += 1
        if gap_counter_rows == step:
            gap_counter_rows = 0

    return torch.from_numpy(init_config_gravity_points.astype(np.int))