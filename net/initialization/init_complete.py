import argparse
import os

from net.initialization.folders.default_folders import default_folders_dict
from net.initialization.folders.experiment_complete_folders import experiment_complete_folders_dict
from net.initialization.path.experiment_complete_result_path import experiment_complete_result_path_dict
from net.initialization.utility.create_folder_and_subfolder import create_folder_and_subfolder


def initialization_complete(network_name: str,
                            experiment_complete_ID: str,
                            parser: argparse.Namespace,
                            debug: bool) -> dict:
    """
    Initialization of experiment complete results folder

    :param network_name: network name
    :param experiment_complete_ID: experiment complete ID
    :param parser: parser of parameters-parsing
    :param debug: debug option
    :return: path dictionary
    """

    # ------------ #
    # FOLDERS DICT #
    # ------------ #
    # default folders
    default_folders = default_folders_dict(where=parser.where)

    # experiment complete folders
    experiment_complete_folders = experiment_complete_folders_dict()

    # --------- #
    # PATH DICT #
    # --------- #
    # experiment complete
    experiment_complete_name = network_name + "|" + experiment_complete_ID
    experiment_complete_path = os.path.join(default_folders['experiments_complete'], experiment_complete_name)

    # experiment complete path
    experiment_complete_results_path = experiment_complete_result_path_dict(experiment_path=experiment_complete_path,
                                                                            experiment_complete_folders=experiment_complete_folders)

    # ------------- #
    # CREATE FOLDER #
    # ------------- #
    # create experiment folder
    if not debug:
        if parser.mode in ['script_test_complete', 'script_detections']:
            # create folder
            create_folder_and_subfolder(main_path=experiment_complete_path,
                                        subfolder_path_dict=experiment_complete_results_path)
            print("Experiment Complete result folder: COMPLETE")

        else:
            print("Experiment Complete result folder: ALREADY COMPLETE")
    else:
        print("Debug Initialization")

    # ---- #
    # PATH #
    # ---- #
    # detections
    detections_test_complete_filename = "detections-test|" + experiment_complete_ID + ".csv"
    detections_test_complete_path = os.path.join(experiment_complete_results_path['detections'], detections_test_complete_filename)

    detections_test_NMS_complete_filename = "detections-test-NMS={}x{}|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_complete_ID + ".csv"
    detections_test_NMS_complete_path = os.path.join(experiment_complete_results_path['detections'], detections_test_NMS_complete_filename)

    # metrics-test
    metrics_test_complete_filename = "metrics-test|" + experiment_complete_ID + ".csv"
    metrics_test_complete_path = os.path.join(experiment_complete_results_path['metrics_test'], metrics_test_complete_filename)

    metrics_test_NMS_complete_filename = "metrics-test-NMS={}x{}|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_complete_ID + ".csv"
    metrics_test_NMS_complete_path = os.path.join(experiment_complete_results_path['metrics_test'], metrics_test_NMS_complete_filename)

    # plots-test
    FROC_test_complete_filename = "FROC|" + experiment_complete_ID + ".png"
    FROC_test_complete_path = os.path.join(experiment_complete_results_path['plots_test'], FROC_test_complete_filename)

    FROC_linear_test_complete_filename = "FROC-Linear|" + experiment_complete_ID + ".png"
    FROC_linear_test_complete_path = os.path.join(experiment_complete_results_path['plots_test'], FROC_linear_test_complete_filename)

    ROC_test_complete_filename = "ROC|" + experiment_complete_ID + ".png"
    ROC_test_complete_path = os.path.join(experiment_complete_results_path['plots_test'], ROC_test_complete_filename)

    score_distribution_test_complete_filename = "Score-Distribution|" + experiment_complete_ID + ".png"
    score_distribution_test_complete_path = os.path.join(experiment_complete_results_path['plots_test'], score_distribution_test_complete_filename)

    # coords test
    FROC_test_complete_coords_filename = "FROC-coords|" + experiment_complete_ID + ".csv"
    FROC_test_complete_coords_path = os.path.join(experiment_complete_results_path['coords_test'], FROC_test_complete_coords_filename)

    ROC_test_complete_coords_filename = "ROC-coords|" + experiment_complete_ID + ".csv"
    ROC_test_complete_coords_path = os.path.join(experiment_complete_results_path['coords_test'], ROC_test_complete_coords_filename)

    # plots-test NMS
    FROC_test_NMS_complete_filename = "FROC-NMS={}x{}|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_complete_ID + ".png"
    FROC_test_NMS_complete_path = os.path.join(experiment_complete_results_path['plots_test_NMS'], FROC_test_NMS_complete_filename)

    FROC_linear_test_NMS_complete_filename = "FROC-Linear-NMS={}x{}|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_complete_ID + ".png"
    FROC_linear_test_NMS_complete_path = os.path.join(experiment_complete_results_path['plots_test_NMS'], FROC_linear_test_NMS_complete_filename)

    ROC_test_NMS_complete_filename = "ROC-NMS={}x{}|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_complete_ID + ".png"
    ROC_test_NMS_complete_path = os.path.join(experiment_complete_results_path['plots_test_NMS'], ROC_test_NMS_complete_filename)

    score_distribution_test_NMS_complete_filename = "Score-Distribution-NMS={}x{}|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_complete_ID + ".png"
    score_distribution_test_NMS_complete_path = os.path.join(experiment_complete_results_path['plots_test_NMS'], score_distribution_test_NMS_complete_filename)

    # coords test NMS
    FROC_test_NMS_complete_coords_filename = "FROC-NMS={}x{}-coords|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_complete_ID + ".csv"
    FROC_test_NMS_complete_coords_path = os.path.join(experiment_complete_results_path['coords_test_NMS'], FROC_test_NMS_complete_coords_filename)

    ROC_test_NMS_complete_coords_filename = "ROC-NMS={}x{}-coords|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_complete_ID + ".csv"
    ROC_test_NMS_complete_coords_path = os.path.join(experiment_complete_results_path['coords_test_NMS'], ROC_test_NMS_complete_coords_filename)

    # path
    path = {
        'detections': {
            'test': detections_test_complete_path,
            'test_NMS': {
                '{}x{}'.format(parser.NMS_box_radius, parser.NMS_box_radius): detections_test_NMS_complete_path,
            },
        },

        'detections_rocalc': experiment_complete_results_path['detections_rocalc'],
        'rocalc_cases': experiment_complete_results_path['detections_rocalc_subfolder']['cases'],
        'rocalc_detections': experiment_complete_results_path['detections_rocalc_subfolder']['detections'],

        'metrics': {
            'test': metrics_test_complete_path,
            'test_NMS': {
                '{}x{}'.format(parser.NMS_box_radius, parser.NMS_box_radius): metrics_test_NMS_complete_path,
            }
        },

        'plots_test': {
            'FROC': FROC_test_complete_path,
            'FROC_linear': FROC_linear_test_complete_path,

            'ROC': ROC_test_complete_path,

            'coords': {
                'FROC': FROC_test_complete_coords_path,
                'ROC': ROC_test_complete_coords_path,
            },

            'score_distribution': score_distribution_test_complete_path,
        },

        'plots_test_NMS': {

            '{}x{}'.format(parser.NMS_box_radius, parser.NMS_box_radius): {
                'FROC': FROC_test_NMS_complete_path,
                'FROC_linear': FROC_linear_test_NMS_complete_path,

                'ROC': ROC_test_NMS_complete_path,

                'coords': {
                    'FROC': FROC_test_NMS_complete_coords_path,
                    'ROC': ROC_test_NMS_complete_coords_path,
                },

                'score_distribution': score_distribution_test_NMS_complete_path,
            }
        }
    }

    return path
