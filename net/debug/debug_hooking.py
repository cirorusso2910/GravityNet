import cv2
import numpy as np
import torch

from net.colors.colors import *
from net.dataset.draw.add_3_channels_image import add_3_channels_image
from net.dataset.utility.viewable_image import viewable_image


def debug_hooking(gravity_points: torch.Tensor,
                  annotation: torch.Tensor,
                  assigned_annotations: torch.Tensor,
                  positive_indices: torch.Tensor,
                  negative_indices: torch.Tensor,
                  rejected_indices: torch.Tensor,
                  image: torch.Tensor,
                  save: bool,
                  path: str):
    """
    DEBUG HOOKING

    save:
        - image with gravity points configuration
          and colors the gravity points hooked with 'similar' color

    :param gravity_points: gravity points configuration
    :param annotation: annotation
    :param assigned_annotations: assigned annotations
    :param positive_indices: gravity points positive indices
    :param negative_indices: gravity points negative indices
    :param rejected_indices: gravity points rejected indices
    :param image: image
    :param save: save option
    :param path: path to save
    """

    # gravity points with positive indices
    gravity_points_positive = gravity_points[positive_indices]
    gravity_points_positive_x = gravity_points_positive[:, 0]
    gravity_points_positive_y = gravity_points_positive[:, 1]

    # gravity points with rejected indices
    gravity_points_rejected = gravity_points[rejected_indices]
    gravity_points_rejected_x = gravity_points_rejected[:, 0]
    gravity_points_rejected_y = gravity_points_rejected[:, 1]

    # annotation assigned to each positive gravity points
    assigned_annotations = assigned_annotations[positive_indices, :]
    annotation_coord_assigned_x = assigned_annotations[:, 0]
    annotation_coord_assigned_y = assigned_annotations[:, 1]

    # num gravity points
    num_gravity_points = gravity_points.shape[0]

    # num gravity points positive
    num_gravity_points_positive = positive_indices.sum()

    # num gravity points rejected
    num_gravity_points_rejected = rejected_indices.sum()

    # num annotations
    num_annotations = annotation.shape[0]

    # draw all gravity points (off)
    for a in range(num_gravity_points):
        # gravity point coord
        coord_gravity_point_x = int(gravity_points[a, 0])
        coord_gravity_point_y = int(gravity_points[a, 1])
        cv2.circle(image, (coord_gravity_point_x, coord_gravity_point_y), radius=0, color=YELLOW1, thickness=-1)

    # draw all annotation (off)
    for t in range(num_annotations):
        # annotation coord
        coord_annotation_x = int(annotation[t, 0])
        coord_annotation_y = int(annotation[t, 1])

        cv2.circle(image, (coord_annotation_x, coord_annotation_y), radius=0, color=RED1, thickness=-1)

    # draw gravity points (positive)
    for p in range(num_gravity_points_positive):
        # positive gravity point coord
        coord_gravity_point_positive_x = int(gravity_points_positive_x[p])
        coord_gravity_point_positive_y = int(gravity_points_positive_y[p])

        # annotation hooked to positive gravity point
        coord_annotation_x = int(annotation_coord_assigned_x[p])
        coord_annotation_y = int(annotation_coord_assigned_y[p])

        # draw annotation hooked
        # for num_t in range(num_annotations):
        #     if annotation_coord_assigned_x[p] == annotation[num_t, 0] and annotation_coord_assigned_y[p] == annotation[num_t, 1]:
        #         cv2.circle(image, (coord_gravity_point_positive_x, coord_gravity_point_positive_y), radius=0, color=color_positive_gravity_points[num_t], thickness=-1)
        #         cv2.circle(image, (coord_annotation_x, coord_annotation_y), radius=1, color=color_annotation[num_t], thickness=-1)
        cv2.circle(image, (coord_gravity_point_positive_x, coord_gravity_point_positive_y), radius=0, color=CYAN2, thickness=-1)
        cv2.circle(image, (coord_annotation_x, coord_annotation_y), radius=0, color=GREEN1, thickness=-1)

    # draw gravity points (rejected)
    for r in range(num_gravity_points_rejected):
        # rejected gravity point coord
        coord_gravity_point_rejected_x = int(gravity_points_rejected_x[r])
        coord_gravity_point_rejected_y = int(gravity_points_rejected_y[r])

        cv2.circle(image, (coord_gravity_point_rejected_x, coord_gravity_point_rejected_y), radius=0, color=YELLOW1, thickness=-1)

    if save:
        # save image
        cv2.imwrite(path, image)
