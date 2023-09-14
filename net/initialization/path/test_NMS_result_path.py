import os


def test_NMS_result_path_dict(experiment_path, experiment_folders):
    """
    Concatenate Test-NMS result path

    :param experiment_path: experiment path
    :param experiment_folders: experiment folders dictionary
    :return: test-NMS result path dictionary
    """

    # detections
    detections_path = os.path.join(experiment_path, experiment_folders['detections'])

    # metrics test
    metrics_test_path = os.path.join(experiment_path, experiment_folders['metrics_test'])

    # output
    output_path = os.path.join(experiment_path, experiment_folders['output'])

    # output test NMS
    output_test_NMS_path = os.path.join(output_path, experiment_folders['output_test_NMS'])

    # output gravity test NMS
    output_gravity_test_NMS_path = os.path.join(output_path, experiment_folders['output_gravity']['test_NMS'])

    # plots test NMS
    plots_test_NMS_path = os.path.join(experiment_path, experiment_folders['plots_test_NMS'])
    coords_test_NMS_path = os.path.join(plots_test_NMS_path, experiment_folders['coords_test_NMS'])

    experiment_result_path = {
        'detections': detections_path,

        'metrics_test': metrics_test_path,

        'output': output_path,

        'output_gravity_test_NMS': output_gravity_test_NMS_path,
        'output_test_NMS': output_test_NMS_path,

        'plots_test_NMS': plots_test_NMS_path,
        'coords_test_NMS': coords_test_NMS_path
    }

    return experiment_result_path
