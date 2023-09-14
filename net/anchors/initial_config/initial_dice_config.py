import numpy as np
import torch


def initial_dice_config(dice: int,
                        image_shape: np.ndarray,
                        feature_map_shape: np.ndarray) -> torch.Tensor:
    """
    Generate initial gravity points with dice configuration in a reference window (feature map)

    :param dice: dice number [1,2,3,4,5]
    :param image_shape: image shape
    :param feature_map_shape: feature map shape
    :return: initial gravity points with dice configuration
    """

    # feature map grid on image (H x W)
    gridFM_height = int(round((image_shape[0] / feature_map_shape[0])))  # 32 (H)
    gridFM_width = int(round((image_shape[1] / feature_map_shape[1])))  # 30 (W)

    # feature map grid rescale
    grid_height = gridFM_height - 16
    grid_width = gridFM_width - 16

    # generate the 'base' anchor box
    base_anchor = np.array([-grid_width * 0.5, grid_height * 0.5, grid_width * 0.5, -grid_height * 0.5])  # [x1, y1, x2, y2]

    # coord base anchor
    x1 = base_anchor[0]  # x1
    y1 = base_anchor[1]  # y1
    x2 = base_anchor[2]  # x2
    y2 = base_anchor[3]  # y2

    coord_top_left = [x1, y1]
    coord_bottom_left = [x1, y2]
    coord_top_right = [x2, y1]
    coord_bottom_right = [x2, y2]
    coord_center = [0, 0]

    # initial configuration
    init_config_gravity_points = np.zeros((0, 2)).astype(np.int)

    # ------ #
    # DICE 1 #
    # ------ #
    #
    #       *
    #
    if dice == 1:
        init_config_gravity_points = np.append(init_config_gravity_points, [coord_center], axis=0)

    # ------ #
    # DICE 2 #
    # ------ #
    #   *
    #
    #           *
    elif dice == 2:
        init_config_gravity_points = np.append(init_config_gravity_points, [coord_top_left], axis=0)
        init_config_gravity_points = np.append(init_config_gravity_points, [coord_bottom_right], axis=0)

    # ------ #
    # DICE 3 #
    # ------ #
    #   *
    #       *
    #           *
    elif dice == 3:
        init_config_gravity_points = np.append(init_config_gravity_points, [coord_top_left], axis=0)
        init_config_gravity_points = np.append(init_config_gravity_points, [coord_center], axis=0)
        init_config_gravity_points = np.append(init_config_gravity_points, [coord_bottom_right], axis=0)

    # ------ #
    # DICE 4 #
    # ------ #
    #   *       *
    #
    #   *       *
    elif dice == 4:
        init_config_gravity_points = np.append(init_config_gravity_points, [coord_bottom_left], axis=0)
        init_config_gravity_points = np.append(init_config_gravity_points, [coord_bottom_right], axis=0)
        init_config_gravity_points = np.append(init_config_gravity_points, [coord_top_left], axis=0)
        init_config_gravity_points = np.append(init_config_gravity_points, [coord_top_right], axis=0)

    # ------ #
    # DICE 5 #
    # ------ #
    #   *       *
    #       *
    #   *       *
    elif dice == 5:
        init_config_gravity_points = np.append(init_config_gravity_points, [coord_top_left], axis=0)
        init_config_gravity_points = np.append(init_config_gravity_points, [coord_bottom_left], axis=0)
        init_config_gravity_points = np.append(init_config_gravity_points, [coord_center], axis=0)
        init_config_gravity_points = np.append(init_config_gravity_points, [coord_top_right], axis=0)
        init_config_gravity_points = np.append(init_config_gravity_points, [coord_bottom_right], axis=0)

    return torch.from_numpy(init_config_gravity_points.astype(np.int))
