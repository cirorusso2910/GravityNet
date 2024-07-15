import sys

from net.utility.msg.msg_error import msg_error


def metrics_dict(metrics_type: str) -> dict:
    """
    Get metrics dictionary according to type

    :param metrics_type: metrics type
    :return: metrics dictionary
    """

    if metrics_type == 'train':
        metrics = {
            'ticks': [],

            'loss': {
                'loss': [],
                'classification': [],
                'regression': []
            },

            'learning_rate': [],

            'AUC': [],

            'sensitivity': {
                '10 FPS': [],
                'max': []
            },

            'AUFROC': {
                '[0, 1]': [],
                '[0, 10]': [],
                '[0, 50]': [],
                '[0, 100]': []
            },

            'AUPR': [],

            'time': {
                'train': [],
                'validation': [],
                'metrics': []
            }
        }

    elif metrics_type == 'test':
        metrics = {
            'AUC': [],

            'sensitivity': {
                '10 FPS': [],
                'max': []
            },

            'AUFROC': {
                '[0, 1]': [],
                '[0, 10]': [],
                '[0, 50]': [],
                '[0, 100]': []
            },

            'AUPR': [],

            'time': {
                'test': [],
                'metrics': []
            }
        }

    else:
        str_err = msg_error(file=__file__,
                            variable=metrics_type,
                            type_variable='metrics type',
                            choices='[train, test]')
        sys.exit(str_err)

    return metrics
