import os
import cv2
import numpy as np

from pandas import read_csv
from torch.utils.data import Dataset

from net.dataset.utility.read_dataset_sample import read_dataset_sample
from net.colors.colors import *
from net.metrics.utility.my_round_value import my_round_value
from net.output.utility.image_tensor_to_numpy import image_tensor_to_numpy
from net.output.utility.select_image_channel import select_image_channel


def output_FPS(score_FPS: float,
               type_draw: str,
               dataset: Dataset,
               num_images: int,
               detections_path: str,
               output_path: str,
               suffix: str):
    """
    Save detections output results at specific false positive per scan (FPS)

    :param score_FPS:
    :param type_draw: type draw
    :param dataset: dataset
    :param num_images: num images
    :param detections_path: detections path
    :param output_path: output path
    :param suffix: suffix
    """

    # read detections test for showing output (numpy array)
    detections = read_csv(filepath_or_buffer=detections_path, usecols=["FILENAME", "LABEL", "SCORE", "PREDICTION X", "PREDICTION Y", "TARGET X", "TARGET Y", "RADIUS"]).dropna(subset=['LABEL']).values

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
        index_subset = np.where(image_filename == detections)[0].tolist()
        detections_subset = detections[index_subset]

        # labels
        labels = detections_subset[:, 1]
        # scores
        scores = detections_subset[:, 2]
        # predictions
        predictions = detections_subset[:, 3:5]
        # annotations detected
        annotations_detected = detections_subset[:, 5:8]
        # num prediction
        num_prediction = len(labels)

        # image conversion
        image = image_tensor_to_numpy(image=sample['image'])

        # select image channel [default: 'RGB']
        image = select_image_channel(image=image,
                                     channel='RGB')

        # ---------------- #
        # DRAW ANNOTATIONS #
        # ---------------- #
        for t in range(num_annotations):

            # annotation
            coord_annotation_x = int(sample['annotation'][t, 0])
            coord_annotation_y = int(sample['annotation'][t, 1])
            radius_annotation = int(sample['annotation'][t, 2].item())

            # draw annotations (circle)
            if type_draw == 'circle':
                cv2.circle(image, (coord_annotation_x, coord_annotation_y), radius=1, color=RED1, thickness=1)

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
            coord_prediction_x = int(predictions[p, 0])
            coord_prediction_y = int(predictions[p, 1])
            radius_prediction = 3  # mean radius

            # draw prediction (FP) over score
            if label == 0 and score > score_FPS:

                # draw prediction (FP) (circle)
                if type_draw == 'circle':
                    cv2.circle(image, (coord_prediction_x, coord_prediction_y), radius=0, color=VIOLET, thickness=-1)
                    cv2.putText(image, text=str(score), org=(coord_prediction_x, coord_prediction_y - 3),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=VIOLET, thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)

                # draw prediction (FP) (bounding box)
                if type_draw == 'box':
                    start_point_annotation = (coord_prediction_x - radius_prediction, coord_prediction_y - radius_prediction)  # start point (top left corner of rectangle)
                    end_point_annotation = (coord_prediction_x + radius_prediction, coord_prediction_y + radius_prediction)  # end point (bottom right corner of rectangle)# draw prediction (TP)
                    cv2.rectangle(image, pt1=start_point_annotation, pt2=end_point_annotation, color=VIOLET, thickness=1)
                    cv2.putText(image, text=str(score), org=(start_point_annotation[0], start_point_annotation[1] - 10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=VIOLET, thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)

            # draw prediction (TP) over score
            elif label == 1 and score > score_FPS:

                # annotation detected
                coord_annotation_detected_x = int(annotations_detected[p, 0])
                coord_annotation_detected_y = int(annotations_detected[p, 1])
                radius_annotation_detected = int(annotations_detected[p, 2])

                if type_draw == 'circle':
                    # draw prediction (TP) (circle)
                    cv2.circle(image, (coord_prediction_x, coord_prediction_y), radius=0, color=BLUE, thickness=-1)

                    # draw annotations detected (circle)
                    cv2.circle(image, (coord_annotation_detected_x, coord_annotation_detected_y), radius=1, color=GREEN1, thickness=1)
                    cv2.putText(image, text=str(score), org=(coord_prediction_x, coord_prediction_y - 3),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=GREEN1, thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)

                if type_draw == 'box':
                    # draw prediction (TP) (bounding box)
                    start_point_annotation = (coord_prediction_x - radius_annotation_detected, coord_prediction_y - radius_annotation_detected)  # start point (top left corner of rectangle)
                    end_point_annotation = (coord_prediction_x + radius_annotation_detected, coord_prediction_y + radius_annotation_detected)  # end point (bottom right corner of rectangle)# draw prediction (TP)
                    cv2.rectangle(image, pt1=start_point_annotation, pt2=end_point_annotation, color=BLUE, thickness=1)

                    # draw annotations detected (bounding box)
                    start_point_annotation = (coord_annotation_detected_x - radius_annotation_detected, coord_annotation_detected_y - radius_annotation_detected)  # start point (top left corner of rectangle)
                    end_point_annotation = (coord_annotation_detected_x + radius_annotation_detected, coord_annotation_detected_y + radius_annotation_detected)  # end point (bottom right corner of rectangle)
                    cv2.rectangle(image, pt1=start_point_annotation, pt2=end_point_annotation, color=GREEN1, thickness=1)
                    cv2.putText(image, text=str(score), org=(start_point_annotation[0], start_point_annotation[1] - 10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=GREEN1, thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)

            # draw prediction (TP) under score
            elif label == 1 and score <= score_FPS:

                # annotation detected underscore
                coord_annotation_detected_x_underscore = int(annotations_detected[p, 0])
                coord_annotation_detected_y_underscore = int(annotations_detected[p, 1])
                radius_annotation_detected_underscore = int(annotations_detected[p, 2])

                if type_draw == 'circle':
                    # draw prediction underscore (TP) (circle)
                    cv2.circle(image, (coord_prediction_x, coord_prediction_y), radius=0, color=ORANGE, thickness=-1)

                    # draw annotations detected underscore (circle)
                    cv2.circle(image, (coord_annotation_detected_x_underscore, coord_annotation_detected_y_underscore), radius=1, color=RED1, thickness=1)

                if type_draw == 'box':
                    # draw prediction (TP) underscore (bounding box)
                    start_point_annotation = (coord_prediction_x - radius_annotation_detected_underscore, coord_prediction_y - radius_annotation_detected_underscore)  # start point (top left corner of rectangle)
                    end_point_annotation = (coord_prediction_x + radius_annotation_detected_underscore, coord_prediction_y + radius_annotation_detected_underscore)  # end point (bottom right corner of rectangle)# draw prediction (TP)
                    cv2.rectangle(image, pt1=start_point_annotation, pt2=end_point_annotation, color=ORANGE, thickness=1)

                    # draw annotations detected underscore (bounding box)
                    start_point_annotation = (coord_annotation_detected_x_underscore - radius_annotation_detected_underscore, coord_annotation_detected_y_underscore - radius_annotation_detected_underscore)  # start point (top left corner of rectangle)
                    end_point_annotation = (coord_annotation_detected_x_underscore + radius_annotation_detected_underscore, coord_annotation_detected_y_underscore + radius_annotation_detected_underscore)  # end point (bottom right corner of rectangle)
                    cv2.rectangle(image, pt1=start_point_annotation, pt2=end_point_annotation, color=RED1, thickness=1)
                    cv2.putText(image, text=str(score), org=(start_point_annotation[0], start_point_annotation[1] - 10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=RED1, thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)

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
