def experiment_folders_dict() -> dict:
    """
    Experiment folders dictionary

    :return: folders dictionary
    """

    experiment_result_folders = {
        'detections': 'detections',
        'log': 'log',
        'metrics_test': 'metrics-test',
        'metrics_train': 'metrics-train',
        'models': 'models',

        'output': 'output',
        'output_test': 'output-test',
        'output_gravity': {
            'validation': 'output-gravity-validation',
            'test': 'output-gravity-test',
        },

        'plots_test': 'plots-test',
        'coords_test': 'coords',

        'plots_train': 'plots-train',

        'plots_validation': 'plots-validation',
        'plots_FROC_validation': 'FROC-validation',
        'plots_ROC_validation': 'ROC-validation',
        'plots_PR_validation': 'PR-validation',

        'coords_validation': 'coords',
        'coords_FROC_validation': 'coords-FROC-validation',
        'coords_ROC_validation': 'coords-ROC-validation',
        'coords_PR_validation': 'coords-PR-validation'

    }

    return experiment_result_folders
