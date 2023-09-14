import numpy as np
import torch

from pandas import read_csv
from torch.utils.data import Dataset

from net.dataset.utility.read_dataset_sample import read_dataset_sample
from net.metrics.metrics_sensitivity_images import metrics_sensitivity_images
from net.metrics.utility.my_round_value import my_round_value


def sensitivity_images(detections_path: str,
                       score_FPS: float,
                       dataset: Dataset,
                       num_images_normals: int,
                       sensitivity_path: str,
                       show: bool,
                       save: bool):
    """
    Compute sensitivity per image

    :param detections_path: detections path
    :param score_FPS: score threshold of specific FPS
    :param dataset: dataset
    :param num_images_normals: num images normals
    :param sensitivity_path: sensitivity path
    :param show: show option
    :param save: save option
    """

    # read detections test (numpy array)
    detections_filename = read_csv(filepath_or_buffer=detections_path, usecols=["FILENAME"]).values
    detections_label = torch.from_numpy(read_csv(filepath_or_buffer=detections_path, usecols=["LABEL"]).values)
    detections_score = torch.from_numpy(read_csv(filepath_or_buffer=detections_path, usecols=["SCORE"]).values)

    # num images (dataset len)
    num_images = dataset.__len__()

    # init
    num_TP_hist = []
    num_FP_hist = []
    num_FN_hist = []
    num_annotations_hist = []

    # for each sample in dataset
    for i in range(num_images):
        # read dataset sample
        sample = read_dataset_sample(dataset=dataset,
                                     idx=i)

        # image filename
        image_filename = sample['filename']

        # num annotations
        num_annotations = sample['annotation'].shape[0]

        # detections subset filename
        index_subset = np.where(image_filename == detections_filename)[0].tolist()

        # label subset
        label_subset = detections_label[index_subset]
        # score subset
        score_subset = detections_score[index_subset]
        # num prediction subset
        num_prediction_subset = len(label_subset)

        # detections subset score
        index_subset_score_gt = torch.gt(score_subset, score_FPS)[:, 0]

        # label subset score
        label_subset_score = label_subset[index_subset_score_gt]
        # num prediction subset score
        num_prediction_subset_score = len(label_subset_score)

        num_TP = label_subset_score[label_subset_score[:, 0] == 1].shape[0]
        num_FP = label_subset_score[label_subset_score[:, 0] == 0].shape[0]
        num_FN = num_annotations - num_TP

        num_TP_hist.append(num_TP)
        num_FP_hist.append(num_FP)
        num_FN_hist.append(num_FN)
        num_annotations_hist.append(num_annotations)

        # sensitivity image
        if num_annotations != 0:
            sensitivity_image = my_round_value((num_TP / num_annotations) * 100, digits=1)
            sens = "{} %".format(sensitivity_image)
        else:
            sens = " "

        # row
        row = [image_filename, num_annotations, num_TP, num_FN, num_FP, sens]

        if show:
            print("Image {}/{}: {} | Annotations: {} | TP: {} | FN: {} | FP: {} | Sensitivity: {}".format(i + 1,
                                                                                                      num_images,
                                                                                                      image_filename,
                                                                                                      num_annotations,
                                                                                                      num_TP,
                                                                                                      num_FN,
                                                                                                      num_FP,
                                                                                                      sens))

        if save:
            # metrics-sensitivity-image.csv
            metrics_sensitivity_images(metrics_path=sensitivity_path,
                                       row=row)

    tot_TP = sum(num_TP_hist)
    tot_FP = sum(num_FP_hist)
    tot_FN = sum(num_FN_hist)
    tot_annotations = sum(num_annotations_hist)

    avg_FP_scan = tot_FP / num_images_normals
    TPR = tot_TP / tot_annotations

    print("\nTPR: {}".format(TPR),
          "\navg FP scan: {}".format(avg_FP_scan))
