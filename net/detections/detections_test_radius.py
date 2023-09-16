import os

import numpy as np
import pandas as pd
import torch

from net.debug.debug_detections import debug_detections
from net.detections.utility.check_index import check_index
from net.detections.utility.conversion_item_list import conversion_item_list
from net.detections.utility.init_detections import init_detections
from net.initialization.header.detections import detections_header
from net.output.output_gravity import output_gravity
from net.utility.read_file import read_file


def detections_test_radius(experiment_ID: str,
                           filenames: torch.Tensor,
                           predictions: torch.Tensor,
                           classifications: torch.Tensor,
                           images: torch.Tensor,
                           masks: torch.Tensor,
                           annotations: torch.Tensor,
                           factor: int,
                           detections_path: str,
                           FP_list_path: str,
                           output_gravity_path: str,
                           device: torch.device,
                           do_output_gravity: bool,
                           debug: bool):
    """
    Compute detections in test with radius metrics and save detections.csv

    DETECTIONS CRITERION:
        - TP: predictions whose distance to the annotation (microaneurysms) is less than radius
              (radius is different for each microaneurysms)

        - possibleTP: predictions that fit the described criterion
                     (among them the one with the highest score is chosen as TP)

        - FP: predictions that do not fit the described criterion
              (FP all images or FP normals images)

        - FN: annotation (microaneurysms) missed

    FALSE POSITIVE REDUCTION:
        gravity points outside the color fundus image mask are not considered

    OUTPUT GRAVITY:
        saves the output-gravity of each image

    :param experiment_ID: experiment ID
    :param filenames: filenames
    :param predictions: predictions
    :param classifications: classifications score
    :param images: images
    :param masks: masks
    :param annotations: annotations
    :param factor: multiplication factor
    :param detections_path: detections path to save
    :param FP_list_path: FP list images path
    :param output_gravity_path: output gravity path
    :param device: device
    :param do_output_gravity: output gravity option
    :param debug: debug option
    """

    # batch size
    batch_size = classifications.shape[0]

    # images FP list
    images_FP_list = read_file(file_path=FP_list_path)

    # for each batch
    for i in range(batch_size):

        # get filename
        filename = filenames[i]

        # get image
        image = images[i]

        # get mask
        mask = masks[i]
        height, width = mask.shape

        # get classification
        score = classifications[i].to(device)  # A x 2

        # get prediction
        prediction = predictions[i].to(device)  # A x 2

        # get annotation
        annotation = annotations[i]
        annotation = annotation[annotation[:, 0] != -1]  # the real annotation annotation (not -1)
        radius = annotation[:, 2].to(device)
        annotation = annotation[:, :2].to(device)  # (x, y)
        # get num annotations
        num_annotations = annotation.shape[0]

        # ---------------------------------- #
        # PREDICTIONS NEGATIVE AND OUT IMAGE #
        # ---------------------------------- #
        # remove predictions negative
        index_prediction_x_negative = torch.lt(prediction[:, 0], 0)  # x < 0
        index_prediction_y_negative = torch.lt(prediction[:, 1], 0)  # y < 0
        index_prediction_negative = torch.logical_or(input=index_prediction_x_negative,
                                                     other=index_prediction_y_negative)
        prediction[index_prediction_negative] = -2
        score[index_prediction_negative] = -2

        # remove predictions out image
        index_prediction_x_out_image = torch.ge(prediction[:, 0], width)  # x > W
        index_prediction_y_out_image = torch.ge(prediction[:, 1], height)  # y > H
        index_prediction_out_image = torch.logical_or(input=index_prediction_x_out_image,
                                                      other=index_prediction_y_out_image)
        prediction[index_prediction_out_image] = -2
        score[index_prediction_out_image] = -2

        # delete prediction and score with index negative and out image
        prediction = prediction[prediction[:, 0] != -2]
        score = score[score[:, 0] != -2]

        # get num predictions
        num_predictions = prediction.shape[0]

        # --------------- #
        # INIT DETECTIONS #
        # --------------- #
        detections = init_detections(num_predictions=num_predictions,
                                     classification=score,
                                     prediction=prediction,
                                     device=device)

        # -------- #
        # DISTANCE #
        # -------- #
        # compute distance between prediction and annotation
        dist = torch.cdist(prediction.float(), annotation.float())  # P x T

        # init hist
        index_TP_hist = []
        annotation_not_detected_hist = []

        # for each annotation
        for t in range(num_annotations):

            # ----------------- #
            # DISTANCE POSITIVE #
            # ----------------- #
            # TRUE: if distance is < radiusTP | FALSE: else
            dist_positive = torch.le(dist[:, t], radius[t] * factor)  # A

            # ----------- #
            # POSSIBLE TP #
            # ----------- #
            # index positive predictions
            index_positive_predictions = dist_positive.nonzero().tolist()

            # check index positive predictions and index TP
            index_positive_predictions = check_index(index_TP=index_TP_hist, index_positive=index_positive_predictions)

            # set label '-1'
            detections[index_positive_predictions, 1] = -1  # label

            # classification score
            classification_positive_predictions = score[index_positive_predictions, 1].tolist()

            # -------------------- #
            # TP (TARGET DETECTED) #
            # -------------------- #
            # select TP with max classification score
            if classification_positive_predictions:
                # max classification score
                max_classification_positive_predictions = max(classification_positive_predictions)
                # index prediction with max classification score
                max_index_classification_positive_predictions = classification_positive_predictions.index(max_classification_positive_predictions)

                # index TP
                index_TP = index_positive_predictions[max_index_classification_positive_predictions]
                index_TP_hist.append(index_TP)

                # coords annotation detected
                coord_x_annotation_detected = int(round(annotation[t, 0].item(), ndigits=3))
                coord_y_annotation_detected = int(round(annotation[t, 1].item(), ndigits=3))
                radius_annotation_detected = int(radius[t])

                # set label '1'
                detections[index_TP, 1] = 1  # label
                detections[index_TP, 5] = coord_x_annotation_detected  # coord x detected
                detections[index_TP, 6] = coord_y_annotation_detected  # coord y detected
                detections[index_TP, 7] = radius_annotation_detected  # radius annotation detected

            # ------------------ #
            # FN (TARGET MISSED) #
            # ------------------ #
            else:

                # coords annotation not detected
                coord_x_annotation_not_detected = int(round(annotation[t, 0].item(), ndigits=3))
                coord_y_annotation_not_detected = int(round(annotation[t, 1].item(), ndigits=3))
                radius_annotation_not_detected = int(radius[t])

                # append coords hist
                # filename | num predictions | label | score | prediction x | prediction y | annotation x | annotation y | radius
                annotation_not_detected_hist.append([filename, np.nan, np.nan, '-inf', np.nan, np.nan, coord_x_annotation_not_detected, coord_y_annotation_not_detected, radius_annotation_not_detected])

        # --------- #
        # FP IMAGES #
        # --------- #
        if filename not in images_FP_list:
            # index hist
            index_FP_hist = torch.eq(detections[:, 1], 0)

            # set label '-3' (FP no normals)
            detections[index_FP_hist, 1] = -3

        # ----------------------------- #
        # MASK FALSE POSITIVE REDUCTION #
        # ----------------------------- #
        # index predictions negative (< 0)  [ check due image value ]
        index_prediction_x_negative = torch.lt(detections[:, 3], 0)  # x < 0
        index_prediction_y_negative = torch.lt(detections[:, 4], 0)  # y < 0
        index_prediction_negative = torch.logical_or(input=index_prediction_x_negative,
                                                     other=index_prediction_y_negative)

        # set label '-2' (out image)
        detections[index_prediction_negative, 1] = -2

        # index predictions out image boundary (HxW)  [ check due image value ]
        index_prediction_x_out_image = torch.ge(detections[:, 3], width)  # x <= W
        index_prediction_y_out_image = torch.ge(detections[:, 4], height)  # y <= H
        index_prediction_out_image = torch.logical_or(input=index_prediction_x_out_image,
                                                      other=index_prediction_y_out_image)

        # set label '-2' (out image)
        detections[index_prediction_out_image, 1] = -2

        # delete predictions with label '-2' (negative & out image)
        detections = detections[detections[:, 1] != -2]

        # index out mask
        mask_value = mask[detections[:, 4].long(), detections[:, 3].long()]
        index_out_mask = torch.not_equal(input=mask_value, other=255.)
        detections[index_out_mask, 1] = -4

        # -------------- #
        # OUTPUT GRAVITY #
        # -------------- #
        if do_output_gravity:
            # draw output gravity
            output_gravity(image=image,
                           annotation=annotation,
                           detections=detections,
                           output_gravity_path=os.path.join(output_gravity_path, "{}-output-gravity|{}.png".format(filename, experiment_ID)))

        # ----- #
        # DEBUG #
        # ----- #
        if debug:
            debug_detections(image=image,
                             annotation=annotation,
                             detections=detections,
                             path="./debug/detections-debug|filename={}.png".format(filename))

        # delete predictions with label '-1' (possibleTP) '-3' (FP no normals) '-4' (out mask)
        detections = detections[detections[:, 1] != -1]
        detections = detections[detections[:, 1] != -3]
        detections = detections[detections[:, 1] != -4]

        # add filename
        detections = detections.tolist()  # convert to list
        for item in detections:
            item.insert(0, filename)
            # item conversion
            conversion_item_list(item=item)

        # add annotations not detected to detections
        detections_complete = detections + annotation_not_detected_hist

        # --------------- #
        # SAVE DETECTIONS #
        # --------------- #
        # detections_np = detections.cpu().detach().numpy()  # convert detections (tensor) to numpy
        detections_np = np.array(detections_complete)  # convert detections (list) to numpy
        detections_csv = pd.DataFrame(detections_np)
        if not os.path.exists(detections_path):
            detections_csv.to_csv(detections_path, mode='a', index=False, header=detections_header(), float_format='%g')  # write header
        else:
            detections_csv.to_csv(detections_path, mode='a', index=False, header=False, float_format='%g')  # write without header
