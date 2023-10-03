import cv2
import numpy as np
import torch

from net.colors.colors import *
from net.dataset.utility.viewable_image import viewable_image
from net.detections.utility.get_single_detection import get_single_detection
from net.output.utility.select_image_channel import select_image_channel


def output_3C_score_gravity(image: torch.Tensor,
                            annotation: torch.Tensor,
                            detections: torch.Tensor,
                            output_gravity_path: str):
    """
    Save detections output score gravity results (for 3-channels-image) for a single image detections

    :param image: image
    :param annotation: annotation
    :param detections: detections
    :param output_gravity_path: output gravity path
    """

    # num annotations
    num_annotations = annotation.shape[0]

    # num predictions
    num_predictions = detections.shape[0]

    # image conversion
    image = image.permute((1, 2, 0))  # permute 3xHxW -> HxWx3
    image = image.cpu().detach().numpy()  # to numpy
    image = viewable_image(image=image)  # viewable

    # select image channel [default: 'G' (green)]
    image = select_image_channel(image=image,
                                 channel='G')

    # image detections
    image_detections = np.zeros((image.shape[0], image.shape[1], 3))

    # box draw radius
    box_draw_radius = 10

    # ---------------- #
    # DRAW ANNOTATIONS #
    # ---------------- #
    for t in range(num_annotations):
        coord_annotation_x = int(round(annotation[t, 0].item(), ndigits=3))
        coord_annotation_y = int(round(annotation[t, 1].item(), ndigits=3))

        # draw box
        # start_point_annotation = (coord_annotation_x - box_draw_radius, coord_annotation_y - box_draw_radius)  # start point (top left corner of rectangle)
        # end_point_annotation = (coord_annotation_x + box_draw_radius, coord_annotation_y + box_draw_radius)  # end point (bottom right corner of rectangle)
        # cv2.rectangle(image_detections, pt1=start_point_annotation, pt2=end_point_annotation, color=YELLOW1, thickness=-1)

        # draw circle
        cv2.circle(image, (coord_annotation_x, coord_annotation_y), radius=2, color=YELLOW1, thickness=-1)

    # ---------------- #
    # DRAW PREDICTIONS #
    # ---------------- #
    for a in range(num_predictions):

        # get single detection
        single_detection = get_single_detection(index=a, detections=detections)

        # score > 0.5
        if single_detection['score'] > 0.5:
            start_point_annotation = (single_detection['prediction_x'] - box_draw_radius, single_detection['prediction_y'] - box_draw_radius)  # start point (top left corner of rectangle)
            end_point_annotation = (single_detection['prediction_x'] + box_draw_radius, single_detection['prediction_y'] + box_draw_radius)  # end point (bottom right corner of rectangle)
            cv2.rectangle(image_detections, pt1=start_point_annotation, pt2=end_point_annotation, color=GREEN1, thickness=-1)

    # ---------- #
    # SAVE IMAGE #
    # ---------- #
    image_to_save = cv2.addWeighted(src1=image_detections,
                                    alpha=0.3,
                                    src2=image,
                                    beta=0.7,
                                    gamma=0)
    cv2.imwrite(output_gravity_path, image_to_save)
