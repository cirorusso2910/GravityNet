def experiment_complete_folders_dict() -> dict:
    """
    Experiment complete folders dictionary

    :return: folders dictionary
    """

    experiment_result_folders = {
        'detections': 'detections',

        'detections_rocalc': 'detections-ROCalc',
        'detections_rocalc_subfolder': {
            'cases': 'cases',
            'detections': 'detections',
        },

        'metrics_test': 'metrics-test',

        'plots_test': 'plots-test',
        'coords_test': 'coords',

        'plots_test_NMS': 'plots-test-NMS',
        'coords_test_NMS': 'coords',
    }

    return experiment_result_folders
