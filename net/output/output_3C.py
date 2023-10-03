import os
import sys

import cv2
import numpy as np

from pandas import read_csv
from torch.utils.data import Dataset

from net.dataset.utility.read_dataset_sample import read_dataset_sample
from net.colors.colors import *
from net.initialization.header.detections import detections_header
from net.metrics.utility.my_round_value import my_round_value
from net.output.utility.select_image_channel import select_image_channel
from net.utility.msg.msg_error import msg_error


def output(type_draw: str,
           eval: str,
           box_draw_radius: int,
           channel: str,
           dataset: Dataset,
           num_images: int,
           detections_path: str,
           output_path: str,
           suffix: str):
    """
    Save detections output results (for 3-channels-image)

    :param type_draw: type draw
    :param eval: evaluation type
    :param box_draw_radius: box draw radius
    :param channel: image channel
    :param dataset: dataset
    :param num_images: num images
    :param detections_path: detections path
    :param output_path: output path
    :param suffix: suffix
    """

    # -------------- #
    # EVAL: DISTANCE #
    # -------------- #
    if 'distance' in eval:

        # read detections test for showing output (numpy array)
        detections = read_csv(filepath_or_buffer=detections_path, usecols=detections_header(eval='distance')).dropna(subset=['LABEL']).values

        # for each sample in dataset
        for i in range(dataset.__len__()):

            # read dataset sample
            sample = read_dataset_sample(dataset=dataset,
                                         idx=i)

            # image filename
            image_filename = sample['filename']

            # num annotations
            num_annotations = sample['annotation'].shape[0]

            # detections subset filename
            index = np.where(image_filename == detections)[0].tolist()
            detections_subset = detections[index]

            # labels
            labels = detections_subset[:, 2]
            # scores
            scores = detections_subset[:, 3]
            # predictions
            predictions = detections_subset[:, 4:6]
            # annotations detected
            annotations_detected = detections_subset[:, 6:8]

            # num prediction
            num_prediction = predictions.shape[0]

            # image
            image = select_image_channel(image=sample['image'],
                                         channel=channel)

            # ---------------- #
            # DRAW ANNOTATIONS #
            # ---------------- #
            for t in range(num_annotations):

                # annotation
                coord_annotation_x = int(sample['annotation'][t, 0])
                coord_annotation_y = int(sample['annotation'][t, 1])

                # draw annotations (circle)
                if type_draw == 'circle':
                    cv2.circle(image, (coord_annotation_x, coord_annotation_y), radius=0, color=RED1, thickness=1)

                # draw annotations (bounding box)
                if type_draw == 'box':
                    start_point_annotation = (coord_annotation_x - box_draw_radius, coord_annotation_y - box_draw_radius)  # start point (top left corner of rectangle)
                    end_point_annotation = (coord_annotation_x + box_draw_radius, coord_annotation_y + box_draw_radius)  # end point (bottom right corner of rectangle)
                    cv2.rectangle(sample['image'], pt1=start_point_annotation, pt2=end_point_annotation, color=RED1, thickness=1)

            # ---------------- #
            # DRAW PREDICTIONS #
            # ---------------- #
            for p in range(num_prediction):

                # label
                label = int(labels[p])

                # score
                score = my_round_value(scores[p], digits=3)

                # predictions
                coord_prediction_x = int(round(predictions[p, 0], ndigits=0))
                coord_prediction_y = int(round(predictions[p, 1], ndigits=0))

                # draw prediction (FP) [SKIPPED]
                # if label == 0:

                # draw prediction (TP)
                if label == 1:

                    # annotation detected
                    coord_annotation_detected_x = int(round(annotations_detected[p, 0], ndigits=3))
                    coord_annotation_detected_y = int(round(annotations_detected[p, 1], ndigits=3))

                    if type_draw == 'circle':
                        # draw prediction (TP) (circle)
                        cv2.circle(image, (coord_prediction_x, coord_prediction_y), radius=0, color=BLUE, thickness=1)

                        # draw annotations detected (circle)
                        cv2.circle(image, (coord_annotation_detected_x, coord_annotation_detected_y), radius=0, color=GREEN1, thickness=1)
                        cv2.putText(image, text=str(score), org=(coord_annotation_detected_x, coord_annotation_detected_y - 3),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=GREEN1, thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)

                    elif type_draw == 'box':
                        # draw prediction (TP) (bounding box)
                        start_point_prediction = (coord_prediction_x - box_draw_radius, coord_prediction_y - box_draw_radius)  # start point (top left corner of rectangle)
                        end_point_prediction = (coord_prediction_x + box_draw_radius, coord_prediction_y + box_draw_radius)  # end point (bottom right corner of rectangle)
                        cv2.rectangle(image, pt1=start_point_prediction, pt2=end_point_prediction, color=BLUE, thickness=1)

                        # draw annotations detected (bounding box)
                        start_point_annotation = (coord_annotation_detected_x - box_draw_radius, coord_annotation_detected_y - box_draw_radius)  # start point (top left corner of rectangle)
                        end_point_annotation = (coord_annotation_detected_x + box_draw_radius, coord_annotation_detected_y + box_draw_radius)  # end point (bottom right corner of rectangle)
                        cv2.rectangle(sample['image'], pt1=start_point_annotation, pt2=end_point_annotation, color=GREEN1, thickness=1)
                        cv2.putText(sample['image'], text=str(score), org=(start_point_annotation[0], start_point_annotation[1] - 10),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=GREEN1, thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)

            # -------- #
            # FILENAME #
            # -------- #
            image_output_filename = image_filename + suffix

            # image output path
            image_output_path = os.path.join(output_path, image_output_filename + ".png")

            # ---------- #
            # SAVE IMAGE #
            # ---------- #
            cv2.imwrite(image_output_path, image)
            print("Image {}/{}: {} saved".format(i + 1, num_images, image_filename))

            if num_images == i + 1:
                return

    # ------------ #
    # EVAL: RADIUS #
    # ------------ #
    elif 'radius' in eval:

        # read detections test for showing output (numpy array)
        detections = read_csv(filepath_or_buffer=detections_path, usecols=detections_header(eval='radius')).dropna(subset=['LABEL']).values

        # for each sample in dataset
        for i in range(dataset.__len__()):

            # read dataset sample
            sample = read_dataset_sample(dataset=dataset,
                                         idx=i)

            # image filename
            image_filename = sample['filename']

            # num annotations
            num_annotations = sample['annotation'].shape[0]

            # detections subset filename
            index = np.where(image_filename == detections)[0].tolist()
            detections_subset = detections[index]

            # labels
            labels = detections_subset[:, 2]
            # scores
            scores = detections_subset[:, 3]
            # predictions
            predictions = detections_subset[:, 4:6]
            # annotations detected
            annotations_detected = detections_subset[:, 6:9]

            # num prediction
            num_prediction = predictions.shape[0]

            # image
            image = select_image_channel(image=sample['image'],
                                         channel=channel)

            # ---------------- #
            # DRAW ANNOTATIONS #
            # ---------------- #
            for t in range(num_annotations):

                # annotation
                coord_annotation_x = int(round(sample['annotation'][t, 0].item(), ndigits=3))
                coord_annotation_y = int(round(sample['annotation'][t, 1].item(), ndigits=3))
                radius_annotation = int(sample['annotation'][t, 2].item())

                # draw annotations (circle)
                if type_draw == 'circle':
                    cv2.circle(image, (coord_annotation_x, coord_annotation_y), radius=0, color=RED1, thickness=1)

                # draw annotations (bounding box)
                if type_draw == 'box':
                    start_point_annotation = (coord_annotation_x - radius_annotation, coord_annotation_y - radius_annotation)  # start point (top left corner of rectangle)
                    end_point_annotation = (coord_annotation_x + radius_annotation, coord_annotation_y + radius_annotation)  # end point (bottom right corner of rectangle)
                    cv2.rectangle(image, pt1=start_point_annotation, pt2=end_point_annotation, color=RED1, thickness=1)

            # ---------------- #
            # DRAW PREDICTIONS #
            # ---------------- #
            for p in range(num_prediction):

                # label
                label = int(labels[p])

                # score
                score = my_round_value(scores[p], digits=3)

                # predictions
                coord_prediction_x = int(round(predictions[p, 0], ndigits=0))
                coord_prediction_y = int(round(predictions[p, 1], ndigits=0))

                # draw prediction (FP) [SKIPPED]
                # if label == 0:

                # draw prediction (TP)
                if label == 1:

                    # annotation detected
                    coord_annotation_detected_x = int(round(annotations_detected[p, 0], ndigits=3))
                    coord_annotation_detected_y = int(round(annotations_detected[p, 1], ndigits=3))
                    radius_annotation_detected = int(annotations_detected[p, 2])

                    if type_draw == 'circle':
                        # draw prediction (TP) (circle)
                        cv2.circle(image, (coord_prediction_x, coord_prediction_y), radius=0, color=BLUE, thickness=1)

                        # draw annotations detected (circle)
                        cv2.circle(image, (coord_annotation_detected_x, coord_annotation_detected_y), radius=0, color=GREEN1, thickness=1)
                        cv2.putText(image, text=str(score), org=(coord_annotation_detected_x, coord_annotation_detected_y - 3),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=GREEN1, thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)

                    elif type_draw == 'box':
                        # draw prediction (TP) (bounding box)
                        start_point_prediction = (coord_prediction_x - radius_annotation_detected, coord_prediction_y - radius_annotation_detected)  # start point (top left corner of rectangle)
                        end_point_prediction = (coord_prediction_x + radius_annotation_detected, coord_prediction_y + radius_annotation_detected)  # end point (bottom right corner of rectangle)
                        cv2.rectangle(image, pt1=start_point_prediction, pt2=end_point_prediction, color=BLUE, thickness=1)

                        # draw annotations detected (bounding box)
                        start_point_annotation = (coord_annotation_detected_x - radius_annotation_detected, coord_annotation_detected_y - radius_annotation_detected)  # start point (top left corner of rectangle)
                        end_point_annotation = (coord_annotation_detected_x + radius_annotation_detected, coord_annotation_detected_y + radius_annotation_detected)  # end point (bottom right corner of rectangle)
                        cv2.rectangle(image, pt1=start_point_annotation, pt2=end_point_annotation, color=GREEN1, thickness=1)
                        cv2.putText(image, text=str(score), org=(start_point_annotation[0], start_point_annotation[1] - 10),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=GREEN1, thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)

            # -------- #
            # FILENAME #
            # -------- #
            image_output_filename = image_filename + suffix

            # image output path
            image_output_path = os.path.join(output_path, image_output_filename + ".png")

            # ---------- #
            # SAVE IMAGE #
            # ---------- #
            cv2.imwrite(image_output_path, image)
            print("Image {}/{}: {} saved".format(i + 1, num_images, image_filename))

            if num_images == i + 1:
                return

    else:
        str_err = msg_error(file=__file__,
                            variable=eval,
                            type_variable="evaluation",
                            choices="[distance, radius]")
        sys.exit(str_err)