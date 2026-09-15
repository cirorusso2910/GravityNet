import argparse
import os

from net.initialization.folders.dataset_folders import dataset_folders_dict
from net.initialization.folders.experiment_folders import experiment_folders_dict
from net.initialization.path.experiment_result_path import experiment_result_path_dict
from net.initialization.utility.create_folder import create_folder
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
    experiment_folders = experiment_folders_dict()

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
    experiment_name = network_name + parser.sep + experiment_ID
    experiment_path = os.path.join(parser.experiments_path, experiment_name)

    # experiment path
    experiment_results_path = experiment_result_path_dict(experiment_path=experiment_path,
                                                          experiment_folders=experiment_folders)

    # explainability path
    explainability_folder = "explainability"
    explainability_path = os.path.join(experiment_path, explainability_folder)

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

    elif parser.mode in ['explainability']:
        create_folder(path=explainability_path)
        print("Explainability result folder: COMPLETE")

    else:
        print("Experiment result folder: ALREADY COMPLETE")

    # ----------- #
    # RESULT PATH #
    # ----------- #
    # detections
    detections_validation_filename = "detections-validation" + parser.sep + experiment_ID + ".csv"
    detections_validation_path = os.path.join(experiment_results_path['detections'], detections_validation_filename)

    detections_test_filename = "detections-test" + parser.sep + experiment_ID + ".csv"
    detections_test_path = os.path.join(experiment_results_path['detections'], detections_test_filename)

    # metrics-train
    metrics_train_filename = "metrics-train" + parser.sep + experiment_ID + ".csv"
    metrics_train_path = os.path.join(experiment_results_path['metrics_train'], metrics_train_filename)

    # metrics-test
    metrics_test_filename = "metrics-test" + parser.sep + experiment_ID + ".csv"
    metrics_test_path = os.path.join(experiment_results_path['metrics_test'], metrics_test_filename)

    # models best
    model_best_sensitivity_10_FPS_filename = network_name + "-best-model-sensitivity-10-FPS" + parser.sep + experiment_ID + ".tar"
    model_best_sensitivity_10_FPS_path = os.path.join(experiment_results_path['models'], model_best_sensitivity_10_FPS_filename)

    model_best_AUFROC_0_10_filename = network_name + "-best-model-AUFROC=[0,10]" + parser.sep + experiment_ID + ".tar"
    model_best_AUFROC_0_10_path = os.path.join(experiment_results_path['models'], model_best_AUFROC_0_10_filename)

    model_best_AUPR_filename = network_name + "-best-model-AUPR" + parser.sep + experiment_ID + ".tar"
    model_best_AUPR_path = os.path.join(experiment_results_path['models'], model_best_AUPR_filename)

    # plots-train
    loss_filename = "Loss" + parser.sep + experiment_ID + ".png"
    loss_path = os.path.join(experiment_results_path['plots_train'], loss_filename)

    learning_rate_filename = "LearningRate" + parser.sep + experiment_ID + ".png"
    learning_rate_path = os.path.join(experiment_results_path['plots_train'], learning_rate_filename)

    # plots-validation
    sensitivity_filename = "Sensitivity" + parser.sep + experiment_ID + ".png"
    sensitivity_path = os.path.join(experiment_results_path['plots_validation'], sensitivity_filename)

    AUC_filename = "AUC" + parser.sep + experiment_ID + ".png"
    AUC_path = os.path.join(experiment_results_path['plots_validation'], AUC_filename)

    AUFROC_filename = "AUFROC" + parser.sep + experiment_ID + ".png"
    AUFROC_path = os.path.join(experiment_results_path['plots_validation'], AUFROC_filename)

    AUPR_filename = "AUPR" + parser.sep + experiment_ID + ".png"
    AUPR_path = os.path.join(experiment_results_path['plots_validation'], AUPR_filename)

    # plots-test
    FROC_test_filename = "FROC" + parser.sep + experiment_ID + ".png"
    FROC_test_path = os.path.join(experiment_results_path['plots_test'], FROC_test_filename)

    FROC_test_linear_filename = "FROC-Linear" + parser.sep + experiment_ID + ".png"
    FROC_test_linear_path = os.path.join(experiment_results_path['plots_test'], FROC_test_linear_filename)

    ROC_test_filename = "ROC" + parser.sep + experiment_ID + ".png"
    ROC_test_path = os.path.join(experiment_results_path['plots_test'], ROC_test_filename)

    score_distribution_filename = "Score-Distribution" + parser.sep + experiment_ID + ".png"
    score_distribution_path = os.path.join(experiment_results_path['plots_test'], score_distribution_filename)

    # coords test
    FROC_test_coords_filename = "FROC-coords" + parser.sep + experiment_ID + ".csv"
    FROC_test_coords_path = os.path.join(experiment_results_path['coords_test'], FROC_test_coords_filename)

    ROC_test_coords_filename = "ROC-coords" + parser.sep + experiment_ID + ".csv"
    ROC_test_coords_path = os.path.join(experiment_results_path['coords_test'], ROC_test_coords_filename)

    path = {
        'experiment_path': experiment_path,

        'dataset': path_dataset_dict,

        'detections': {
            'validation': detections_validation_path,
            'test': detections_test_path,
        },

        'explainability': explainability_path,

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
