import argparse
import os

from net.initialization.folders.default_folders import default_folders_dict
from net.initialization.folders.experiment_folders import experiment_folders_dict
from net.initialization.path.experiment_result_path import experiment_result_path_dict
from net.utility.check_file_exists import check_file_exists


def detections_fold_path(network_name: str,
                         experiment_1_fold_ID: str,
                         experiment_2_fold_ID: str,
                         parser: argparse.Namespace) -> dict:
    """
    Concatenate detections 1-fold and 2-fold path

    :param network_name: network name
    :param experiment_1_fold_ID: experiment-1-fold-ID
    :param experiment_2_fold_ID: experiment-2-fold-ID
    :param parser: parser of parameters-parsing
    :return: detections fold path dictionary
    """

    # ------------ #
    # FOLDERS DICT #
    # ------------ #
    # default folders
    default_folders = default_folders_dict(where=parser.where)

    # experiment folders
    experiment_folders = experiment_folders_dict(parser=parser)

    # ---------------- #
    # 1-FOLD PATH DICT #
    # ---------------- #
    # experiment 1-fold
    experiment_1_fold_name = network_name + "|" + experiment_1_fold_ID
    experiment_1_fold_path = os.path.join(default_folders['experiments'], experiment_1_fold_name)

    # experiment 1-fold path
    experiment_1_fold_results_path = experiment_result_path_dict(experiment_path=experiment_1_fold_path,
                                                                 experiment_folders=experiment_folders)

    # detections test 1-fold
    detections_test_1_fold_filename = "detections-test|" + experiment_1_fold_ID + ".csv"
    detections_test_1_fold_path = os.path.join(experiment_1_fold_results_path['detections'], detections_test_1_fold_filename)
    # check_file_exists(path=detections_test_1_fold_path, filename='detections-test 1-fold')

    # detections test NMS 1-fold
    detections_test_NMS_1_fold_filename = "detections-test-NMS={}x{}|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_1_fold_ID + ".csv"
    detections_test_NMS_1_fold_path = os.path.join(experiment_1_fold_results_path['detections'], detections_test_NMS_1_fold_filename)
    # check_file_exists(path=detections_test_NMS_1_fold_path, filename='detections-test NMS-{}x{} 1-fold'.format(parser.NMS_box_radius, parser.NMS_box_radius))

    # ---------------- #
    # 2-FOLD PATH DICT #
    # ---------------- #
    # experiment 2-fold
    experiment_2_fold_name = network_name + "|" + experiment_2_fold_ID
    experiment_2_fold_path = os.path.join(default_folders['experiments'], experiment_2_fold_name)

    # experiment 2-fold path
    experiment_2_fold_results_path = experiment_result_path_dict(experiment_path=experiment_2_fold_path,
                                                                 experiment_folders=experiment_folders)

    # detections 2-fold
    detections_test_2_fold_filename = "detections-test|" + experiment_2_fold_ID + ".csv"
    detections_test_2_fold_path = os.path.join(experiment_2_fold_results_path['detections'], detections_test_2_fold_filename)
    check_file_exists(path=detections_test_2_fold_path, filename='detections-test 2-fold')

    # detections test NMS 3x3 2-fold
    detections_test_NMS_3x3_2_fold_filename = "detections-test-NMS={}x{}|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_2_fold_ID + ".csv"
    detections_test_NMS_3x3_2_fold_path = os.path.join(experiment_2_fold_results_path['detections'], detections_test_NMS_3x3_2_fold_filename)
    check_file_exists(path=detections_test_NMS_3x3_2_fold_path, filename='detections-test NMS-{}x{} 2-fold'.format(parser.NMS_box_radius, parser.NMS_box_radius))

    detections_fold_dict = {
        '1-fold': {
            'test': detections_test_1_fold_path,
            'test_NMS': {
                '{}x{}'.format(parser.NMS_box_radius, parser.NMS_box_radius): detections_test_NMS_1_fold_path,
            }
        },

        '2-fold': {
            'test': detections_test_2_fold_path,
            'test_NMS': {
                '{}x{}'.format(parser.NMS_box_radius, parser.NMS_box_radius): detections_test_NMS_3x3_2_fold_path,
            }
        }
    }

    return detections_fold_dict
