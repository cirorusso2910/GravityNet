import argparse
import os

from net.initialization.folders.dataset_folders import dataset_folders_dict
from net.initialization.folders.experiment_folders import experiment_folders_dict
from net.initialization.path.experiment_result_path import experiment_result_path_dict
from net.initialization.utility.create_folder_and_subfolder import create_folder_and_subfolder


def initialization(network_name: str,
                   experiment_ID: str,
                   parser: argparse.Namespace
                   ) -> dict:
    """
    Initialization of experiment results folder based on execution mode

    :param network_name: network name
    :param experiment_ID: experiment-ID
    :param parser: parser of parameters-parsing
    :return: path dictionary
    """

    # ------------ #
    # FOLDERS DICT #
    # ------------ #
    # dataset folders
    dataset_folders = dataset_folders_dict()

    # experiment folders
    experiment_folders = experiment_folders_dict(parser=parser)

    # ------- #
    # DATASET #
    # ------- #
    # annotations
    annotations_path = os.path.join(parser.dataset_path, parser.dataset, dataset_folders['annotations'], dataset_folders['annotations_subfolder']['csv'], dataset_folders['annotations_subfolder']['csv_subfolder']['all'])

    # images
    images_path = os.path.join(parser.dataset_path, parser.dataset, dataset_folders['images'], dataset_folders['images_subfolder']['all'])

    # images masks
    images_masks_path = os.path.join(parser.dataset_path, parser.dataset, dataset_folders['images'], dataset_folders['images_subfolder']['masks'])

    # data split
    data_split_path = os.path.join(parser.dataset_path, parser.dataset, dataset_folders['split'], 'split-' + parser.split + '.csv')

    # lists
    lists_path = os.path.join(parser.dataset_path, parser.dataset, dataset_folders['lists'])
    list_normals_path = os.path.join(str(lists_path), 'normals.txt')
    list_all_path = os.path.join(str(lists_path), 'all.txt')

    # statistics
    statistics_filename = "split-{}-statistics.csv".format(parser.split)
    rescale_folder = "rescale={}".format(parser.rescale)
    statistics_path = os.path.join(parser.dataset_path, parser.dataset, dataset_folders['statistics'], rescale_folder, statistics_filename)

    # info
    info_path = os.path.join(parser.dataset_path, parser.dataset, dataset_folders['info'])

    path_dataset_dict = {
        'annotations': {
            'all': annotations_path,
        },

        'images': {
            'all': images_path,

            'masks': images_masks_path,
        },

        'lists': {
            'all': list_all_path,
            'normals': list_normals_path,
        },

        'split': data_split_path,

        'statistics': statistics_path,

        'info': info_path
    }

    # ---- #
    # PATH #
    # ---- #
    # experiment
    experiment_name = network_name + "|" + experiment_ID
    experiment_path = os.path.join(parser.experiments_path, experiment_name)

    # experiment path
    experiment_results_path = experiment_result_path_dict(experiment_path=experiment_path,
                                                          experiment_folders=experiment_folders)

    # -------------------- #
    # CREATE RESULT FOLDER #
    # -------------------- #
    # create experiment folder
    if parser.mode in ['train', 'train_test']:
        create_folder_and_subfolder(main_path=experiment_path,
                                    subfolder_path_dict=experiment_results_path)
        print("Experiment result folder: COMPLETE")

    elif parser.mode in ['test']:
        print("Experiment result folder: ALREADY COMPLETE")

    else:
        print("Experiment result folder: ALREADY COMPLETE")

    # ----------- #
    # RESULT PATH #
    # ----------- #
    # detections
    detections_validation_filename = "detections-validation|" + experiment_ID + ".csv"
    detections_validation_path = os.path.join(experiment_results_path['detections'], detections_validation_filename)

    detections_test_filename = "detections-test|" + experiment_ID + ".csv"
    detections_test_path = os.path.join(experiment_results_path['detections'], detections_test_filename)

    # metrics-train
    metrics_train_filename = "metrics-train|" + experiment_ID + ".csv"
    metrics_train_path = os.path.join(experiment_results_path['metrics_train'], metrics_train_filename)

    # metrics-test
    metrics_test_filename = "metrics-test|" + experiment_ID + ".csv"
    metrics_test_path = os.path.join(experiment_results_path['metrics_test'], metrics_test_filename)

    # models best
    model_best_sensitivity_10_FPS_filename = network_name + "-best-model-sensitivity-10-FPS|" + experiment_ID + ".tar"
    model_best_sensitivity_10_FPS_path = os.path.join(experiment_results_path['models'], model_best_sensitivity_10_FPS_filename)

    model_best_AUFROC_0_10_filename = network_name + "-best-model-AUFROC|" + experiment_ID + ".tar"
    model_best_AUFROC_0_10_path = os.path.join(experiment_results_path['models'], model_best_AUFROC_0_10_filename)

    model_best_AUPR_filename = network_name + "-best-model-AUPR|" + experiment_ID + ".tar"
    model_best_AUPR_path = os.path.join(experiment_results_path['models'], model_best_AUPR_filename)

    # plots-train
    loss_filename = "Loss|" + experiment_ID + ".png"
    loss_path = os.path.join(experiment_results_path['plots_train'], loss_filename)

    learning_rate_filename = "LearningRate|" + experiment_ID + ".png"
    learning_rate_path = os.path.join(experiment_results_path['plots_train'], learning_rate_filename)

    # plots-validation
    sensitivity_filename = "Sensitivity|" + experiment_ID + ".png"
    sensitivity_path = os.path.join(experiment_results_path['plots_validation'], sensitivity_filename)

    AUC_filename = "AUC|" + experiment_ID + ".png"
    AUC_path = os.path.join(experiment_results_path['plots_validation'], AUC_filename)

    AUFROC_filename = "AUFROC|" + experiment_ID + ".png"
    AUFROC_path = os.path.join(experiment_results_path['plots_validation'], AUFROC_filename)

    AUPR_filename = "AUPR|" + experiment_ID + ".png"
    AUPR_path = os.path.join(experiment_results_path['plots_validation'], AUPR_filename)

    # plots-test
    FROC_test_filename = "FROC|" + experiment_ID + ".png"
    FROC_test_path = os.path.join(experiment_results_path['plots_test'], FROC_test_filename)

    FROC_test_linear_filename = "FROC-Linear|" + experiment_ID + ".png"
    FROC_test_linear_path = os.path.join(experiment_results_path['plots_test'], FROC_test_linear_filename)

    ROC_test_filename = "ROC|" + experiment_ID + ".png"
    ROC_test_path = os.path.join(experiment_results_path['plots_test'], ROC_test_filename)

    score_distribution_filename = "Score-Distribution|" + experiment_ID + ".png"
    score_distribution_path = os.path.join(experiment_results_path['plots_test'], score_distribution_filename)

    # coords test
    FROC_test_coords_filename = "FROC-coords|" + experiment_ID + ".csv"
    FROC_test_coords_path = os.path.join(experiment_results_path['coords_test'], FROC_test_coords_filename)

    ROC_test_coords_filename = "ROC-coords|" + experiment_ID + ".csv"
    ROC_test_coords_path = os.path.join(experiment_results_path['coords_test'], ROC_test_coords_filename)

    path = {
        'dataset': path_dataset_dict,

        'detections': {
            'validation': detections_validation_path,
            'test': detections_test_path,
        },

        'metrics': {
            'train': metrics_train_path,
            'test': metrics_test_path,
        },

        'model': {
            'best': {
                'sensitivity': {
                    '10 FPS': model_best_sensitivity_10_FPS_path,
                },
                'AUFROC': {
                    '[0, 10]': model_best_AUFROC_0_10_path,
                },
                'AUPR': model_best_AUPR_path,
            },
        },

        'output': {
            'test': experiment_results_path['output_test'],

            'gravity': {
                'validation': experiment_results_path['output_gravity_validation'],
                'test': experiment_results_path['output_gravity_test'],
            }
        },

        'plots_train': {
            'loss': loss_path,
            'learning_rate': learning_rate_path,
        },

        'plots_validation': {
            'sensitivity': sensitivity_path,

            'AUC': AUC_path,

            'AUFROC': AUFROC_path,

            'FROC': experiment_results_path['FROC_validation'],
            'coords_FROC': experiment_results_path['coords_FROC_validation'],

            'ROC': experiment_results_path['ROC_validation'],
            'coords_ROC': experiment_results_path['coords_ROC_validation'],

            'PR': experiment_results_path['PR_validation'],
            'coords_PR': experiment_results_path['coords_PR_validation'],

            'AUPR': AUPR_path,

        },

        'plots_test': {
            'FROC': FROC_test_path,
            'FROC_linear': FROC_test_linear_path,
            'ROC': ROC_test_path,

            'coords': {
                'FROC': FROC_test_coords_path,
                'ROC': ROC_test_coords_path,
            },

            'score_distribution': score_distribution_path,
        }
    }

    return path
