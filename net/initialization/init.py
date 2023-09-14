import argparse
import os
import sys

from net.initialization.folders.EophthaMA_dataset_folders import EophthaMA_dataset_folders_dict
from net.initialization.folders.INbreast_dataset_folders import INbreast_dataset_folders_dict
from net.initialization.folders.default_folders import default_folders_dict
from net.initialization.folders.experiment_folders import experiment_folders_dict
from net.initialization.path.experiment_result_path import experiment_result_path_dict
from net.initialization.path.test_NMS_result_path import test_NMS_result_path_dict
from net.initialization.utility.create_folder import create_folder
from net.initialization.utility.create_folder_and_subfolder import create_folder_and_subfolder
from net.utility.msg.msg_error import msg_error


def initialization(network_name: str,
                   experiment_ID: str,
                   experiment_resume_ID: str,
                   parser: argparse.Namespace,
                   debug: bool) -> dict:
    """
    Initialization of experiment results folder based on execution mode

    :param network_name: network name
    :param experiment_ID: experiment-ID
    :param experiment_resume_ID: experiment-ID for resume
    :param parser: parser of parameters-parsing
    :param debug: debug option
    :return: path dictionary
    """

    # ------------ #
    # FOLDERS DICT #
    # ------------ #
    # default folders
    default_folders = default_folders_dict(where=parser.where)

    # INbreast dataset folders (GravityNet-microcalcifications)
    INbreast_dataset_folders = INbreast_dataset_folders_dict()

    # E-ophtha-MA dataset folders (GravityNet-microaneurysms)
    EophthaMA_dataset_folders = EophthaMA_dataset_folders_dict()

    # experiment folders
    experiment_folders = experiment_folders_dict(parser=parser)

    # ------- #
    # DATASET #
    # ------- #
    # INbreast
    if parser.dataset == 'INbreast':
        # annotations
        annotations_all_path = os.path.join(default_folders['datasets'], parser.dataset, INbreast_dataset_folders['annotations'], INbreast_dataset_folders['annotations_subfolder']['csv'], INbreast_dataset_folders['annotations_subfolder']['csv_subfolder']['all'])
        annotations_cropped_path = os.path.join(default_folders['datasets'], parser.dataset, INbreast_dataset_folders['annotations'], INbreast_dataset_folders['annotations_subfolder']['csv'], INbreast_dataset_folders['annotations_subfolder']['csv_subfolder']['calcifications_cropped'])
        # annotations w48m14 (for training)
        annotations_all_w48m14_path = os.path.join(default_folders['datasets'], parser.dataset, INbreast_dataset_folders['annotations'], INbreast_dataset_folders['annotations_subfolder']['csv'], INbreast_dataset_folders['annotations_subfolder']['csv_subfolder']['calcifications_w48m14'])
        annotations_w48m14_cropped_path = os.path.join(default_folders['datasets'], parser.dataset, INbreast_dataset_folders['annotations'], INbreast_dataset_folders['annotations_subfolder']['csv'], INbreast_dataset_folders['annotations_subfolder']['csv_subfolder']['calcifications_w48m14_cropped'])

        # images
        images_all_path = os.path.join(default_folders['datasets'], parser.dataset, INbreast_dataset_folders['images'], INbreast_dataset_folders['images_subfolder']['all'])
        images_all_cropped_path = os.path.join(default_folders['datasets'], parser.dataset, INbreast_dataset_folders['images'], INbreast_dataset_folders['images_subfolder']['all_cropped'])
        images_all_w48m14_cropped_path = os.path.join(default_folders['datasets'], parser.dataset, INbreast_dataset_folders['images'], INbreast_dataset_folders['images_subfolder']['all_w48m14_cropped'])

        # images masks
        images_masks_path = os.path.join(default_folders['datasets'], parser.dataset, INbreast_dataset_folders['images'], INbreast_dataset_folders['images_subfolder']['masks'])
        images_masks_cropped_path = os.path.join(default_folders['datasets'], parser.dataset, INbreast_dataset_folders['images'], INbreast_dataset_folders['images_subfolder']['masks_cropped'])
        images_masks_w48m14_cropped_path = os.path.join(default_folders['datasets'], parser.dataset, INbreast_dataset_folders['images'], INbreast_dataset_folders['images_subfolder']['masks_w48m14_cropped'])

        # data split
        data_split_path = os.path.join(default_folders['datasets'], parser.dataset, INbreast_dataset_folders['split'], 'split-' + parser.split + '.csv')

        # lists
        lists_path = os.path.join(default_folders['datasets'], parser.dataset, INbreast_dataset_folders['lists'])
        list_normals_path = os.path.join(lists_path, 'normals.txt')
        list_all_path = os.path.join(lists_path, 'all.txt')

        # statistics
        statistics_filename = "split-{}-statistics.csv".format(parser.split)
        statistics_path = os.path.join(default_folders['datasets'], parser.dataset, INbreast_dataset_folders['statistics'], "rescale={}".format(parser.rescale), statistics_filename)

        # info
        info_path = os.path.join(default_folders['datasets'], parser.dataset, INbreast_dataset_folders['info'])

        path_dataset_dict = {
            'annotations': {
                'all': annotations_all_path,
                'cropped': annotations_cropped_path,
                'w48m14': annotations_all_w48m14_path,
                'w48m14_cropped': annotations_w48m14_cropped_path,

                'masks': '',
            },

            'images': {
                'all': images_all_path,
                'cropped': images_all_cropped_path,
                'w48m14_cropped': images_all_w48m14_cropped_path,

                'masks': images_masks_path,
                'masks_cropped': images_masks_cropped_path,
                'masks_w48m14_cropped': images_masks_w48m14_cropped_path,

            },

            'lists': {
                'all': list_all_path,
                'normals': list_normals_path,
            },

            'split': data_split_path,

            'statistics': statistics_path,

            'info': info_path,

        }

    # E-ophtha-MA
    elif parser.dataset == 'E-ophtha-MA':
        # annotations
        annotations_all_path = os.path.join(default_folders['datasets'], parser.dataset, EophthaMA_dataset_folders['annotations'], EophthaMA_dataset_folders['annotations_subfolder']['csv'], EophthaMA_dataset_folders['annotations_subfolder']['csv_subfolder']['all'])
        annotations_cropped_path = os.path.join(default_folders['datasets'], parser.dataset, EophthaMA_dataset_folders['annotations'], EophthaMA_dataset_folders['annotations_subfolder']['csv'], EophthaMA_dataset_folders['annotations_subfolder']['csv_subfolder']['cropped'])
        annotations_resized_path = os.path.join(default_folders['datasets'], parser.dataset, EophthaMA_dataset_folders['annotations'], EophthaMA_dataset_folders['annotations_subfolder']['csv'], EophthaMA_dataset_folders['annotations_subfolder']['csv_subfolder']['resized'])

        # images
        images_all_path = os.path.join(default_folders['datasets'], parser.dataset, EophthaMA_dataset_folders['images'], EophthaMA_dataset_folders['images_subfolder']['all'])
        images_cropped_path = os.path.join(default_folders['datasets'], parser.dataset, EophthaMA_dataset_folders['images'], EophthaMA_dataset_folders['images_subfolder']['cropped'])
        images_green_path = os.path.join(default_folders['datasets'], parser.dataset, EophthaMA_dataset_folders['images'], EophthaMA_dataset_folders['images_subfolder']['green'])
        images_resized_path = os.path.join(default_folders['datasets'], parser.dataset, EophthaMA_dataset_folders['images'], EophthaMA_dataset_folders['images_subfolder']['resized'])

        # images masks
        images_masks_path = os.path.join(default_folders['datasets'], parser.dataset, EophthaMA_dataset_folders['images'], EophthaMA_dataset_folders['images_subfolder']['masks'], EophthaMA_dataset_folders['images_subfolder']['masks_subfolder']['all'])
        images_masks_cropped_path = os.path.join(default_folders['datasets'], parser.dataset, EophthaMA_dataset_folders['images'], EophthaMA_dataset_folders['images_subfolder']['masks'], EophthaMA_dataset_folders['images_subfolder']['masks_subfolder']['cropped'])

        # data split
        data_split_path = os.path.join(default_folders['datasets'], parser.dataset, EophthaMA_dataset_folders['split'], 'split-' + parser.split + '.csv')

        # lists
        lists_path = os.path.join(default_folders['datasets'], parser.dataset, EophthaMA_dataset_folders['lists'])
        list_healthy_path = os.path.join(lists_path, 'healthy.txt')
        list_all_path = os.path.join(lists_path, 'all.txt')

        # statistics
        statistics_filename = "split-{}-statistics.csv".format(parser.split)
        statistics_path = os.path.join(default_folders['datasets'], parser.dataset, EophthaMA_dataset_folders['statistics'], "rescale={}".format(parser.rescale), statistics_filename)

        # info
        info_path = os.path.join(default_folders['datasets'], parser.dataset, EophthaMA_dataset_folders['info'])

        path_dataset_dict = {
            'annotations': {
                'all': annotations_all_path,
                'cropped': annotations_cropped_path,
                'resized': annotations_resized_path
            },

            'images': {
                'all': images_all_path,
                'cropped': images_cropped_path,
                'green': images_green_path,
                'resized': images_resized_path,

                'masks': images_masks_path,
                'masks_cropped': images_masks_cropped_path,
            },

            'lists': {
                'all': list_all_path,
                'normals': list_healthy_path,
            },

            'info': info_path,

            'split': data_split_path,

            'statistics': statistics_path,
        }

    else:
        str_err = msg_error(file=__file__,
                            variable=parser.dataset,
                            type_variable="dataset name",
                            choices="[INbreast, E-ophtha-MA]")
        sys.exit(str_err)

    # ---- #
    # PATH #
    # ---- #
    # experiment
    experiment_name = network_name + "|" + experiment_ID
    experiment_path = os.path.join(default_folders['experiments'], experiment_name)

    # experiment resume
    experiment_resume_name = network_name + "|" + experiment_resume_ID
    experiment_resume_results_path = os.path.join(default_folders['experiments'], experiment_resume_name)

    # experiment path
    experiment_results_path = experiment_result_path_dict(experiment_path=experiment_path,
                                                          experiment_folders=experiment_folders)

    # experiment resume path
    experiment_resume_results_path = experiment_result_path_dict(experiment_path=experiment_resume_results_path,
                                                                 experiment_folders=experiment_folders)

    # test NMS path
    test_NMS_results_path = test_NMS_result_path_dict(experiment_path=experiment_path,
                                                      experiment_folders=experiment_folders)

    # metrics sensitivity path
    metrics_sensitivity_path = os.path.join(experiment_path, experiment_folders['metrics_sensitivity'])

    # output FPS path
    output_FPS_folder = "output-test-FPS={}".format(parser.FPS)
    output_FPS_path = os.path.join(experiment_results_path['output'], output_FPS_folder)

    # output FPS NMS path
    output_FPS_NMS_folder = "output-test-NMS={}x{}-FPS={}".format(parser.NMS_box_radius, parser.NMS_box_radius, parser.FPS)
    output_FPS_NMS_path = os.path.join(experiment_results_path['output'], output_FPS_NMS_folder)

    # -------------------- #
    # CREATE RESULT FOLDER #
    # -------------------- #
    # create experiment folder
    if not debug:
        if parser.mode in ['train', 'resume', 'train_test']:
            create_folder_and_subfolder(main_path=experiment_path,
                                        subfolder_path_dict=experiment_results_path)
            print("Experiment result folder: COMPLETE")

        elif parser.mode in ['test']:
            print("Experiment result folder: ALREADY COMPLETE")

        elif parser.mode in ['test_NMS']:
            create_folder_and_subfolder(main_path=experiment_path,
                                        subfolder_path_dict=test_NMS_results_path)
            print("Experiment NMS result folder: COMPLETE")

        elif parser.mode in ['output_FPS']:
            if parser.do_NMS:
                create_folder(path=output_FPS_NMS_path)
                print("Output FPS NMS result folder: COMPLETE")
            else:
                create_folder(path=output_FPS_path)
                print("Output FPS result folder: COMPLETE")

        elif parser.mode in ['sensitivity_FPS']:
            if parser.do_NMS:
                create_folder(path=metrics_sensitivity_path)
                print("Sensitivity FPS NMS result folder: COMPLETE")
            else:
                create_folder(path=metrics_sensitivity_path)
                print("Sensitivity FPS result folder: COMPLETE")

        else:
            print("Experiment result folder: ALREADY COMPLETE")

    else:
        print("Debug Initialization")

    # ----------- #
    # RESULT PATH #
    # ----------- #
    # detections
    detections_validation_filename = "detections-validation|" + experiment_ID + ".csv"
    detections_validation_path = os.path.join(experiment_results_path['detections'], detections_validation_filename)

    detections_test_filename = "detections-test|" + experiment_ID + ".csv"
    detections_test_path = os.path.join(experiment_results_path['detections'], detections_test_filename)

    detections_test_NMS_filename = "detections-test-NMS={}x{}|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_ID + ".csv"
    detections_test_NMS_path = os.path.join(test_NMS_results_path['detections'], detections_test_NMS_filename)

    # metrics-train
    metrics_train_filename = "metrics-train|" + experiment_ID + ".csv"
    metrics_train_path = os.path.join(experiment_results_path['metrics_train'], metrics_train_filename)

    metrics_train_resume_filename = "metrics-train|" + experiment_resume_ID + ".csv"
    metrics_train_resume_path = os.path.join(experiment_resume_results_path['metrics_train'], metrics_train_resume_filename)

    # metrics-test
    metrics_test_filename = "metrics-test|" + experiment_ID + ".csv"
    metrics_test_path = os.path.join(experiment_results_path['metrics_test'], metrics_test_filename)

    metrics_test_NMS_filename = "metrics-test-NMS={}x{}|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_ID + ".csv"
    metrics_test_NMS_path = os.path.join(test_NMS_results_path['metrics_test'], metrics_test_NMS_filename)

    # metrics sensitivity
    metrics_sensitivity_FPS_filename = "metrics-sensitivity-images-FPS={}|".format(parser.FPS) + experiment_ID + ".csv"
    metrics_sensitivity_FPS_path = os.path.join(metrics_sensitivity_path, metrics_sensitivity_FPS_filename)

    metrics_sensitivity_FPS_NMS_filename = "metrics-sensitivity-images-NMS={}x{}-FPS={}|".format( parser.NMS_box_radius, parser.NMS_box_radius, parser.FPS) + experiment_ID + ".csv"
    metrics_sensitivity_FPS_NMS_path = os.path.join(metrics_sensitivity_path, metrics_sensitivity_FPS_NMS_filename)

    # models best
    model_best_sensitivity_work_point_filename = network_name + "-best-model-sensitivity|" + experiment_ID + ".tar"
    model_best_sensitivity_work_point_path = os.path.join(experiment_results_path['models'], model_best_sensitivity_work_point_filename)

    model_best_AUFROC_0_10_filename = network_name + "-best-model-AUFROC|" + experiment_ID + ".tar"
    model_best_AUFROC_0_10_path = os.path.join(experiment_results_path['models'], model_best_AUFROC_0_10_filename)

    # models resume
    model_resume_filename = network_name + "-resume-model|" + experiment_ID + ".tar"
    model_resume_path = os.path.join(experiment_results_path['models'], model_resume_filename)

    model_resume_to_load_filename = network_name + "-resume-model|" + experiment_resume_ID + ".tar"
    model_resume_to_load_path = os.path.join(experiment_resume_results_path['models'], model_resume_to_load_filename)

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

    # plots-test NMS
    FROC_test_NMS_filename = "FROC-NMS={}x{}|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_ID + ".png"
    FROC_test_NMS_path = os.path.join(test_NMS_results_path['plots_test_NMS'], FROC_test_NMS_filename)

    FROC_test_NMS_linear_filename = "FROC-Linear-NMS={}x{}|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_ID + ".png"
    FROC_test_NMS_linear_path = os.path.join(test_NMS_results_path['plots_test_NMS'], FROC_test_NMS_linear_filename)

    ROC_test_NMS_filename = "ROC-NMS={}x{}|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_ID + ".png"
    ROC_test_NMS_path = os.path.join(test_NMS_results_path['plots_test_NMS'], ROC_test_NMS_filename)

    score_distribution_NMS_filename = "Score-Distribution-NMS={}x{}|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_ID + ".png"
    score_distribution_NMS_path = os.path.join(test_NMS_results_path['plots_test_NMS'], score_distribution_NMS_filename)

    # coords test NMS
    FROC_test_NMS_coords_filename = "FROC-NMS={}x{}-coords|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_ID + ".csv"
    FROC_test_NMS_coords_path = os.path.join(test_NMS_results_path['coords_test_NMS'], FROC_test_NMS_coords_filename)

    ROC_test_NMS_coords_filename = "ROC-NMS={}x{}-coords|".format(parser.NMS_box_radius, parser.NMS_box_radius) + experiment_ID + ".csv"
    ROC_test_NMS_coords_path = os.path.join(test_NMS_results_path['coords_test_NMS'], ROC_test_NMS_coords_filename)

    path = {
        'dataset': path_dataset_dict,

        'detections': {
            'validation': detections_validation_path,
            'test': detections_test_path,
            'test_NMS': detections_test_NMS_path
        },

        'metrics': {
            'train': metrics_train_path,
            'resume': metrics_train_resume_path,
            'test': metrics_test_path,
            'test_NMS': metrics_test_NMS_path,
        },

        'metrics_sensitivity': {
            'FPS': metrics_sensitivity_FPS_path,
            'FPS_NMS': metrics_sensitivity_FPS_NMS_path,
        },

        'model': {
            'best': {
                'sensitivity': model_best_sensitivity_work_point_path,
                'AUFROC': model_best_AUFROC_0_10_path,
            },

            'resume': model_resume_path,
            'resume_to_load': model_resume_to_load_path
        },

        'output': {
            'test': experiment_results_path['output_test'],
            'test_NMS': test_NMS_results_path['output_test_NMS'],

            'FPS': output_FPS_path,
            'FPS_NMS': output_FPS_NMS_path,

            'gravity': {
                'validation': experiment_results_path['output_gravity_validation'],
                'test': experiment_results_path['output_gravity_test'],
                'test_NMS': test_NMS_results_path['output_gravity_test_NMS']
            },

            'resume': {
                'gravity': {
                    'validation': experiment_resume_results_path['output_gravity_validation'],
                }
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

            'resume': {
                'FROC': experiment_resume_results_path['FROC_validation'],
                'coords_FROC': experiment_resume_results_path['coords_FROC_validation'],

                'ROC': experiment_resume_results_path['ROC_validation'],
                'coords_ROC': experiment_resume_results_path['coords_ROC_validation'],
            },
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
        },

        'plots_test_NMS': {
            'FROC': FROC_test_NMS_path,
            'FROC_linear': FROC_test_NMS_linear_path,
            'ROC': ROC_test_NMS_path,

            'coords': {
                'FROC': FROC_test_NMS_coords_path,
                'ROC': ROC_test_NMS_coords_path,
            },

            'score_distribution': score_distribution_NMS_path,
        },

    }

    return path
