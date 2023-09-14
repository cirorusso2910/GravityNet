import cv2
import sys
import torch

from net.colors.colors import *
from net.detections.utility.get_single_detection import get_single_detection
from net.output.utility.image_tensor_to_numpy import image_tensor_to_numpy


def debug_detections(image: torch.Tensor,
                     annotation: torch.Tensor,
                     detections: torch.Tensor,
                     path: str):
    """
    DEBUG DETECTIONS

    save:
        - single image detections (in ./debug folder)

    :param image: image
    :param annotation: annotation
    :param detections: detections
    :param path: path to save
    """

    # image tensor conversion
    image = image_tensor_to_numpy(image=image)

    # num annotations
    num_annotations = annotation.shape[0]
    # print("num annotations: ", num_annotations)

    # num detections
    num_detections = detections.shape[0]

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
    for a in range(num_detections):

        # get single detection
        single_detection = get_single_detection(index=a,
                                                detections=detections)

        # draw FP
        if single_detection['label'] == 0:
            cv2.circle(image, (single_detection['prediction_x'], single_detection['prediction_y']), radius=0, color=GRAY, thickness=-1)

        # draw possibleTP
        elif single_detection['label'] == -1:
            cv2.circle(image, (single_detection['prediction_x'], single_detection['prediction_y']), radius=0, color=CYAN2, thickness=-1)

        # draw negative & out image
        elif single_detection['label'] == -2:
            cv2.circle(image, (single_detection['prediction_x'], single_detection['prediction_y']), radius=0, color=BLACK, thickness=-1)

        # draw FP no normals
        elif single_detection['label'] == -3:
            cv2.circle(image, (single_detection['prediction_x'], single_detection['prediction_y']), radius=0, color=VIOLET, thickness=-1)

        # draw out mask
        elif single_detection['label'] == -4:
            cv2.circle(image, (single_detection['prediction_x'], single_detection['prediction_y']), radius=0, color=YELLOW1, thickness=-1)

        # draw TP
        elif single_detection['label'] == 1:
            cv2.circle(image, (single_detection['prediction_x'], single_detection['prediction_y']), radius=0, color=BLUE, thickness=-1)
            cv2.circle(image, (single_detection['annotation_x'], single_detection['annotation_y']), radius=0, color=GREEN1, thickness=-1)

    # save image
    cv2.imwrite(path, image)

    sys.exit("\nDEBUG DETECTIONS DISTANCE: COMPLETE")
