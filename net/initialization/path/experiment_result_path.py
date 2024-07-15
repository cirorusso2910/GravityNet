import os


def experiment_result_path_dict(experiment_path: str,
                                experiment_folders: dict) -> dict:
    """
    Concatenate experiment result path

    :param experiment_path: experiment path
    :param experiment_folders: experiment folders dictionary
    :return: experiment complete result path dictionary
    """

    # detections
    detections_path = os.path.join(experiment_path, experiment_folders['detections'])

    # log
    log_path = os.path.join(experiment_path, experiment_folders['log'])

    # metrics test
    metrics_test_path = os.path.join(experiment_path, experiment_folders['metrics_test'])

    # metrics train
    metrics_train_path = os.path.join(experiment_path, experiment_folders['metrics_train'])

    # models
    models_path = os.path.join(experiment_path, experiment_folders['models'])

    # output
    output_path = os.path.join(experiment_path, experiment_folders['output'])

    # output gravity validation
    output_gravity_validation_path = os.path.join(output_path, experiment_folders['output_gravity']['validation'])

    # output test
    output_test_path = os.path.join(output_path, experiment_folders['output_test'])

    # output gravity test
    output_gravity_test_path = os.path.join(output_path, experiment_folders['output_gravity']['test'])

    # plots test
    plots_test_path = os.path.join(experiment_path, experiment_folders['plots_test'])
    coords_test_path = os.path.join(plots_test_path, experiment_folders['coords_test'])

    # plots train
    plots_train_path = os.path.join(experiment_path, experiment_folders['plots_train'])

    # plots validation
    plots_validation_path = os.path.join(experiment_path, experiment_folders['plots_validation'])
    coords_validation_path = os.path.join(plots_validation_path, experiment_folders['coords_validation'])

    FROC_validation_path = os.path.join(plots_validation_path, experiment_folders['plots_FROC_validation'])
    coords_FROC_validation_path = os.path.join(coords_validation_path, experiment_folders['coords_FROC_validation'])

    ROC_validation_path = os.path.join(plots_validation_path, experiment_folders['plots_ROC_validation'])
    coords_ROC_validation_path = os.path.join(coords_validation_path, experiment_folders['coords_ROC_validation'])

    PR_validation_path = os.path.join(plots_validation_path, experiment_folders['plots_PR_validation'])
    coords_PR_validation_path = os.path.join(coords_validation_path, experiment_folders['coords_PR_validation'])

    experiment_result_path = {
        'detections': detections_path,

        'log': log_path,

        'metrics_test': metrics_test_path,
        'metrics_train': metrics_train_path,
        'models': models_path,

        'output': output_path,

        'output_gravity_validation': output_gravity_validation_path,
        'output_gravity_test': output_gravity_test_path,
        'output_test': output_test_path,

        'plots_test': plots_test_path,
        'coords_test': coords_test_path,

        'plots_train': plots_train_path,

        'plots_validation': plots_validation_path,
        'coords_validation': coords_validation_path,

        'FROC_validation': FROC_validation_path,
        'coords_FROC_validation': coords_FROC_validation_path,

        'ROC_validation': ROC_validation_path,
        'coords_ROC_validation': coords_ROC_validation_path,

        'PR_validation': PR_validation_path,
        'coords_PR_validation': coords_PR_validation_path,
    }

    return experiment_result_path
