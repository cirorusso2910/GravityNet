import cv2
import torch

from net.colors.colors import *
from net.detections.utility.get_single_detection import get_single_detection


def output_gravity(image: torch.Tensor,
                   annotation: torch.Tensor,
                   detections: torch.Tensor,
                   output_gravity_path: str):
    """
    Save detections output gravity results for a single image detections

    :param image: image
    :param annotation: annotation
    :param detections: detections
    :param output_gravity_path: output gravity path
    """

    # num annotations
    num_annotations = annotation.shape[0]

    # num predictions
    num_predictions = detections.shape[0]

    # ---------------- #
    # DRAW ANNOTATIONS #
    # ---------------- #
    for t in range(num_annotations):
        coord_annotation_x = int(round(annotation[t, 0].item(), ndigits=3))
        coord_annotation_y = int(round(annotation[t, 1].item(), ndigits=3))

        cv2.circle(image, (coord_annotation_x, coord_annotation_y), radius=0, color=RED1, thickness=-1)

    # ---------------- #
    # DRAW PREDICTIONS #
    # ---------------- #
    for a in range(num_predictions):

        # get single detection
        single_detection = get_single_detection(index=a, detections=detections)

        # draw FP
        if single_detection['label'] == 0:
            cv2.circle(image, (single_detection['prediction_x'], single_detection['prediction_y']), radius=0, color=VIOLET, thickness=-1)

        # draw possibleTP
        elif single_detection['label'] == -1:
            cv2.circle(image, (single_detection['prediction_x'], single_detection['prediction_y']), radius=0, color=CYAN2, thickness=-1)

        # draw negative & out mask & FP no normals
        elif single_detection['label'] == -2 or single_detection['label'] == -3 or single_detection['label'] == -4:
            cv2.circle(image, (single_detection['prediction_x'], single_detection['prediction_y']), radius=0, color=YELLOW1, thickness=-1)

        # draw TP
        elif single_detection['label'] == 1:
            cv2.circle(image, (single_detection['prediction_x'], single_detection['prediction_y']), radius=0, color=BLUE, thickness=-1)
            cv2.circle(image, (single_detection['annotation_x'], single_detection['annotation_y']), radius=0, color=GREEN1, thickness=-1)

    # ---------- #
    # SAVE IMAGE #
    # ---------- #
    cv2.imwrite(output_gravity_path, image)
