import cv2
import torch

from net.colors.colors import *


def draw_gravity_points(image: torch.Tensor,
                        annotation: torch.Tensor,
                        gravity_points: torch.Tensor,
                        output_path: str):
    """
    Draw gravity points configuration on image and save

    :param image: image
    :param annotation: annotation
    :param gravity_points: gravity points
    :param output_path: path to save
    """

    # delete ground truth padding
    annotation = annotation[annotation[:, 0] != -1]  # delete padding

    # draw each gravity points
    for gp in gravity_points:
        coord_x = int(gp[0])  # x
        coord_y = int(gp[1])  # y

        # draw gravity point (x, y)
        cv2.circle(image, (coord_x, coord_y), radius=0, color=YELLOW1, thickness=1)

    for annotation in annotation:
        x = int(annotation[0].item())
        y = int(annotation[1].item())

        cv2.circle(image, (x, y), radius=0, color=RED1, thickness=1)

    # save image
    cv2.imwrite(output_path, image)
