import argparse


def experiment_folders_dict(parser: argparse.Namespace) -> dict:
    """
    Experiment folders dictionary

    :param parser: parser of parameters-parsing
    :return: folders dictionary
    """

    experiment_result_folders = {
        'detections': 'detections',
        'log': 'log',
        'metrics_sensitivity': 'metrics-sensitivity',
        'metrics_test': 'metrics-test',
        'metrics_train': 'metrics-train',
        'models': 'models',

        'output': 'output',
        'output_test': 'output-test',
        'output_test_NMS': 'output-test-NMS={}x{}'.format(parser.NMS_box_radius, parser.NMS_box_radius),
        'output_gravity': {
            'validation': 'output-gravity-validation',
            'test': 'output-gravity-test',
            'test_NMS': 'output-gravity-test-NMS={}x{}'.format(parser.NMS_box_radius, parser.NMS_box_radius),
        },

        'plots_test': 'plots-test',
        'coords_test': 'coords',

        'plots_test_NMS': 'plots-test-NMS',
        'coords_test_NMS': 'coords',

        'plots_train': 'plots-train',

        'plots_validation': 'plots-validation',
        'plots_FROC_validation': 'FROC-validation',
        'plots_ROC_validation': 'ROC-validation',

        'coords_validation': 'coords',
        'coords_FROC_validation': 'coords-FROC-validation',
        'coords_ROC_validation': 'coords-ROC-validation',

    }

    return experiment_result_folders
