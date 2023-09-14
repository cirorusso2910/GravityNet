import numpy as np
import torch


def shift(feature_map_shape: np.ndarray,
          stride: int,
          gravity_initial_config: torch.Tensor) -> torch.Tensor:
    """
    Shift initial configuration over the whole image:
    build a shift meshgrid and shift the initial configuration in a reference window (feature map)
    over the whole image

    :param feature_map_shape: feature map shape
    :param stride: stride
    :param gravity_initial_config: gravity points initial configuration
    :return: gravity points configuration
    """

    # compute shift
    shift_x = (np.arange(0, feature_map_shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, feature_map_shape[0]) + 0.5) * stride

    # meshgrid
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # shift horizontal

    # shifts
    # start -> | -> | -> | ->
    #       -> | -> | -> | ->
    #       -> | -> | -> | -> | ... end
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),  # (x, y)
    )).transpose()

    A = gravity_initial_config.shape[0]  # A
    K = shifts.shape[0]  # K shifts

    all_anchors = (gravity_initial_config.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))

    all_anchors = all_anchors.reshape((K * A, 2))  # K * A = num gravity points (image)

    return all_anchors
