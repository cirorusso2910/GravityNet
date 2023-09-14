import os


def experiment_complete_result_path_dict(experiment_path: str,
                                         experiment_complete_folders: dict) -> dict:
    """
    Concatenate experiment complete result path

    :param experiment_path: experiment path
    :param experiment_complete_folders: experiment complete folders dictionary
    :return: experiment complete result path dictionary
    """

    # detections
    detections_path = os.path.join(experiment_path, experiment_complete_folders['detections'])

    # detections rocalc
    detections_rocalc_path = os.path.join(experiment_path, experiment_complete_folders['detections_rocalc'])
    rocalc_cases_path = os.path.join(detections_rocalc_path, experiment_complete_folders['detections_rocalc_subfolder']['cases'])
    rocalc_detections_path = os.path.join(detections_rocalc_path, experiment_complete_folders['detections_rocalc_subfolder']['detections'])

    # metrics test
    metrics_test_path = os.path.join(experiment_path, experiment_complete_folders['metrics_test'])

    # plots test
    plots_test_path = os.path.join(experiment_path, experiment_complete_folders['plots_test'])
    coords_test_path = os.path.join(plots_test_path, experiment_complete_folders['coords_test'])

    # plots test NMS
    plots_test_NMS_path = os.path.join(experiment_path, experiment_complete_folders['plots_test_NMS'])
    coords_test_NMS_path = os.path.join(plots_test_NMS_path, experiment_complete_folders['coords_test_NMS'])

    experiment_result_path = {
        'detections': detections_path,

        'detections_rocalc': detections_rocalc_path,
        'detections_rocalc_subfolder': {
            'cases': rocalc_cases_path,
            'detections': rocalc_detections_path,
        },

        'metrics_test': metrics_test_path,

        'plots_test': plots_test_path,
        'coords_test': coords_test_path,

        'plots_test_NMS': plots_test_NMS_path,
        'coords_test_NMS': coords_test_NMS_path,
    }

    return experiment_result_path
