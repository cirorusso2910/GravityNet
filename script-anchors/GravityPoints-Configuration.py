import numpy as np
import torch

from net.anchors.gravity_points_config import gravity_points_config
from net.anchors.utility.check_image_shape import check_image_shape

from net.debug.debug_anchors import debug_anchors
from net.parameters.parameters import parameters_parsing
from net.utility.execution_mode import execution_mode


def main():
    """
        | ------------------------------ |
        | GRAVITY POINTS - CONFIGURATION |
        | ------------------------------ |

        Show gravity-points details and save configuration:
            - save gravity-points initial configuration
            - save gravity-points complete configuration
            - save Feature Grid

        Path to save: doc/gravity-points-configuration

    """

    print("| ------------------------------ |\n"
          "| GRAVITY POINTS - CONFIGURATION |\n"
          "| ------------------------------ |\n")

    # ------------------ #
    # PARAMETERS-PARSING #
    # ------------------ #
    # command line parameter parsing
    parser = parameters_parsing()

    # execution mode start
    execution_mode(mode=parser.mode,
                   option='start')

    # ------ #
    # DEVICE #
    # ------ #
    print("\n-------"
          "\nDEVICE:"
          "\n-------")
    device = torch.device("cpu")  # device cpu
    print("Device: {}".format(device))

    # -------------- #
    # GRAVITY POINTS #
    # -------------- #
    print("\n---------------"
          "\nGRAVITY POINTS:"
          "\n---------------")
    # image shape (H x W)
    image_shape = np.array((int(parser.image_height), int(parser.image_width)))  # converts to numpy.array

    # check image shape dimension
    check_image_shape(image_shape=image_shape)

    # generate gravity points
    gravity_points, gravity_points_feature_map, feature_map_shape = gravity_points_config(config=parser.config,
                                                                                          image_shape=image_shape,
                                                                                          device=device)

    num_gravity_points = gravity_points.shape[0]  # num anchors points (A)
    num_gravity_points_feature_map = gravity_points_feature_map.shape[0]  # num anchors points in a feature map (reference window)

    print("\nConfig: {}".format(parser.config),
          "\nImage shape: {} x {}".format(image_shape[0], image_shape[1]),
          "\nFeature Map shape: {} x {}".format(feature_map_shape[0], feature_map_shape[1]),
          "\nGravity Points for image: {}".format(num_gravity_points),
          "\nGravity Points for feature map: {}".format(num_gravity_points_feature_map))

    # ----------- #
    # SAVE CONFIG #
    # ----------- #
    if parser.save_config:

        # save anchors
        debug_anchors(config=parser.config,
                      image_shape=image_shape,
                      gravity_points_initial_config=gravity_points_feature_map,
                      gravity_points=gravity_points)

        print("\nSave Gravity Points Configuration: Complete")

    # execution mode complete
    execution_mode(mode=parser.mode,
                   option='complete')


if __name__ == '__main__':
    main()
