import os.path

from pandas import read_csv

from net.initialization.header.metrics import metrics_header
from net.metrics.utility.my_round_value import my_round_value


def read_metrics_test_csv(metrics_path: str) -> dict:
    """
    Read metrics-test.csv

    :param metrics_path: metrics path
    :return: metrics dictionary
    """

    # read metrics
    header = metrics_header(metrics_type='test')
    if os.path.exists(metrics_path):
        metrics_test = read_csv(filepath_or_buffer=metrics_path, usecols=header)
        print("metrics-test.csv reading: COMPLETE")

        # metrics round
        AUC = my_round_value(value=metrics_test['AUC'][0], digits=3)
        sensitivity_work_point = my_round_value(value=metrics_test['SENSITIVITY WORK POINT'][0], digits=3)
        sensitivity_max = my_round_value(value=metrics_test['SENSITIVITY MAX'][0], digits=3)
        AUFROC_0_1 = my_round_value(value=metrics_test['AUFROC [0, 1]'][0], digits=3)
        AUFROC_0_10 = my_round_value(value=metrics_test['AUFROC [0, 10]'][0], digits=3)
        AUFROC_0_50 = my_round_value(value=metrics_test['AUFROC [0, 50]'][0], digits=3)
        AUFROC_0_100 = my_round_value(value=metrics_test['AUFROC [0, 100]'][0], digits=3)

    else:
        AUC = ""
        sensitivity_work_point = ""
        sensitivity_max = ""
        AUFROC_0_1 = ""
        AUFROC_0_10 = ""
        AUFROC_0_50 = ""
        AUFROC_0_100 = ""

    # performance metrics test
    metrics = {
        'AUC': AUC,
        'sensitivity': {
            'work_point': sensitivity_work_point,
            'max': sensitivity_max,
        },
        'AUFROC': {
            '[0, 1]': AUFROC_0_1,
            '[0, 10]': AUFROC_0_10,
            '[0, 50]': AUFROC_0_50,
            '[0, 100]': AUFROC_0_100
        }
    }

    return metrics
