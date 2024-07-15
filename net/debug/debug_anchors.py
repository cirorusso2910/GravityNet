import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame

from net.anchors.utility.get_feature_grid import get_feature_grid
from net.colors.random_color import random_colors
from net.initialization.utility.create_folder import create_folder


def debug_anchors(image_shape: np.ndarray,
                  config: str,
                  gravity_points_initial_config: torch.Tensor,
                  gravity_points: torch.Tensor):
    """
    DEBUG ANCHORS

    save:
        - initial gravity points configuration (image)
        - initial gravity points configuration (csv)

        - gravity points configuration (image)
        - gravity points configuration (csv)

        - feature grid (image)

    :param image_shape: image shape
    :param config: configuration
    :param gravity_points_initial_config: gravity points initial configuration
    :param gravity_points: gravity points configuration
    """

    # num gravity points
    num_gravity_points = gravity_points.shape[0]

    # num gravity points feature map
    num_gravity_points_feature_map = gravity_points_initial_config.shape[0]

    # init result folder
    result_folder_filename = "GravityPoints-Configuration|config={}".format(config)
    doc_folder = "./doc/gravity-points-configuration"
    result_folder_path = os.path.join(doc_folder, result_folder_filename)
    create_folder(path=result_folder_path)
    result_folder_csv_path = os.path.join(result_folder_path, "csv")
    create_folder(path=result_folder_csv_path)
    result_folder_image_path = os.path.join(result_folder_path, "image")
    create_folder(path=result_folder_image_path)

    # init result path
    gravity_points_initial_config_image_filename_pdf = "GravityPoints-InitialConfiguration|config={}|gravity-points-per-feature-map={}.pdf".format(config, num_gravity_points_feature_map)
    gravity_points_initial_config_image_path_pdf = os.path.join(result_folder_image_path, gravity_points_initial_config_image_filename_pdf)

    gravity_points_initial_config_image_filename_png = "GravityPoints-InitialConfiguration|config={}|gravity-points-per-feature-map={}.png".format(config, num_gravity_points_feature_map)
    gravity_points_initial_config_image_path_png = os.path.join(result_folder_image_path, gravity_points_initial_config_image_filename_png)

    gravity_points_initial_config_template_image_filename = "{}-template.pdf".format(config)
    gravity_points_initial_config_template_image_path = os.path.join(result_folder_image_path, gravity_points_initial_config_template_image_filename)

    gravity_points_initial_config_csv_filename = "GravityPoints-InitialConfiguration|config={}|gravity-points-per-feature-map={}.csv".format(config, num_gravity_points_feature_map)
    gravity_points_initial_config_csv_path = os.path.join(result_folder_csv_path, gravity_points_initial_config_csv_filename)

    gravity_points_config_image_filename = "GravityPoints|config={}|gravity-points={}.png".format(config, num_gravity_points)
    gravity_points_config_image_path = os.path.join(result_folder_image_path, gravity_points_config_image_filename)

    gravity_points_config_csv_filename = "GravityPoints|config={}|gravity-points={}.csv".format(config, num_gravity_points)
    gravity_points_config_csv_path = os.path.join(result_folder_csv_path, gravity_points_config_csv_filename)

    feature_grid_filename = "FeatureGrid.png"
    feature_grid_path = os.path.join(result_folder_path, feature_grid_filename)

    # ------------------------------------------ #
    # save initial gravity points config (image) #
    # ------------------------------------------ #
    save_initial_gravity_points_config_image(title="INITIAL GRAVITY POINTS CONFIGURATION",
                                             config=config,
                                             gravity_points=gravity_points_initial_config,
                                             path=gravity_points_initial_config_image_path_pdf)

    save_initial_gravity_points_config_image(title="INITIAL GRAVITY POINTS CONFIGURATION",
                                             config=config,
                                             gravity_points=gravity_points_initial_config,
                                             path=gravity_points_initial_config_image_path_png)

    save_initial_gravity_points_config_template_image(title=config,
                                                      gravity_points=gravity_points_initial_config,
                                                      path=gravity_points_initial_config_template_image_path)

    # ---------------------------------------- #
    # save initial gravity points config (csv) #
    # ---------------------------------------- #
    save_gravity_points_config_csv(gravity_points=gravity_points_initial_config,
                                   path=gravity_points_initial_config_csv_path)

    # ---------------------------------- #
    # save gravity points config (image) #
    # ---------------------------------- #
    save_gravity_points_config_image(image_shape=image_shape,
                                     gravity_points=gravity_points,
                                     num_gravity_points_feature_map=num_gravity_points_feature_map,
                                     path=gravity_points_config_image_path)

    # -------------------------------- #
    # save gravity points config (csv) #
    # -------------------------------- #
    save_gravity_points_config_csv(gravity_points=gravity_points,
                                   path=gravity_points_config_csv_path)

    # ------------------------- #
    # save feature grid (image) #
    # ------------------------- #
    save_feature_grid_image(image_shape=image_shape,
                            path=feature_grid_path)


def save_initial_gravity_points_config_image(title: str,
                                             config: str,
                                             gravity_points: torch.Tensor,
                                             path: str):
    """
    Save initial gravity points configuration (image)

    :param title: title of plot
    :param config: configuration
    :param gravity_points: gravity points configuration
    :param path: path to save
    """

    # save image
    fig = plt.figure(figsize=(6, 6))
    plt.suptitle("{}".format(title), fontweight="bold", fontsize=18)
    plt.title("config: {}".format(config), fontsize=11, pad=10, loc='center')
    plt.scatter(gravity_points[:, 0], gravity_points[:, 1], color='blue', marker='.', s=30)
    plt.xlabel("x")
    plt.xticks(np.arange(-15, 20, step=5))
    plt.ylabel("y")
    plt.yticks(np.arange(-15, 20, step=5))
    plt.savefig(path, bbox_inches='tight')
    plt.clf()  # clear figure
    plt.close(fig)


def save_initial_gravity_points_config_template_image(title: str,
                                                      gravity_points: torch.Tensor,
                                                      path: str):
    """
    Save initial gravity points configuration (template)

    :param title: title of plot
    :param gravity_points: gravity points configuration
    :param path: path to save
    """

    # save image
    fig = plt.figure(figsize=(6, 6))
    # plt.title("{}".format(title), fontweight="bold", fontsize=18)
    plt.scatter(gravity_points[:, 0], gravity_points[:, 1], color='blue', marker='.', s=30)
    plt.xlabel("K", fontsize=35, labelpad=30)
    plt.xticks([])
    plt.ylabel("K", fontsize=35, rotation='horizontal', labelpad=30)
    plt.yticks([])
    plt.savefig(path, bbox_inches='tight')
    plt.clf()  # clear figure
    plt.close(fig)


def save_gravity_points_config_csv(gravity_points: torch.Tensor,
                                   path: str):
    """
    Save gravity points configuration (csv)

    :param gravity_points: gravity points
    :param path: path to save
    """

    # save csv
    DataFrame(gravity_points).to_csv(path, mode='w', header=["X", "Y"], index=False, float_format='%d')


def save_gravity_points_config_image(image_shape: np.ndarray,
                                     gravity_points: torch.Tensor,
                                     num_gravity_points_feature_map: int,
                                     path: str):
    """
    Save gravity points configuration (image)

    :param image_shape: image shape
    :param gravity_points: gravity points configuration
    :param num_gravity_points_feature_map: num gravity points for feature map
    :param path: path to save
    """

    # init image
    gravity_points_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    # init draw gravity
    i = 0  # init index
    bgr = random_colors()  # init random colors

    # draw each gravity
    for gp in gravity_points:
        coord_x = int(gp[0])  # x
        coord_y = int(gp[1])  # y

        if i == num_gravity_points_feature_map:
            bgr = random_colors()  # new random colors
            i = 0  # reset

        # draw gravity point (x, y)
        cv2.circle(gravity_points_image, (coord_x, coord_y), radius=0, color=bgr, thickness=1)

        i = i + 1  # index + 1

    # save image
    cv2.imwrite(path, gravity_points_image)


def save_feature_grid_image(image_shape: np.ndarray,
                            path: str):
    """
    Save feature grid image

    :param image_shape: image shape
    :param path: path to save
    :return:
    """

    # image shape (H x W)
    image_shape = np.array(image_shape)

    # init image
    feature_grid_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    # feature grid line
    shift_height, shift_width = get_feature_grid(image_height=image_shape[0],
                                                 image_width=image_shape[1])

    for w in shift_width:
        cv2.line(feature_grid_image, (int(w), 0), (int(w), image_shape[0]), color=(128, 128, 128,), thickness=1)

    for h in shift_height:
        cv2.line(feature_grid_image, (0, int(h)), (image_shape[1], int(h)), color=(128, 128, 128,), thickness=1)

    # save image
    cv2.imwrite(path, feature_grid_image)
