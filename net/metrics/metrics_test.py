import csv

from net.initialization.header.metrics import metrics_header
from net.metrics.utility.my_round_value import my_round_value
from net.metrics.utility.timer import timer


def metrics_test_csv(metrics_path: str,
                     metrics: dict):
    """
    Save metrics-test.csv

    :param metrics_path: metrics path
    :param metrics: metrics dictionary
    """

    # metrics round
    AUC = my_round_value(value=metrics['AUC'][-1], digits=3)
    sensitivity_10_FPS = my_round_value(value=metrics['sensitivity']['10 FPS'][-1], digits=3)
    sensitivity_max = my_round_value(value=metrics['sensitivity']['max'][-1], digits=3)
    AUFROC_0_1 = my_round_value(value=metrics['AUFROC']['[0, 1]'][-1], digits=3)
    AUFROC_0_10 = my_round_value(value=metrics['AUFROC']['[0, 10]'][-1], digits=3)
    AUFROC_0_50 = my_round_value(value=metrics['AUFROC']['[0, 50]'][-1], digits=3)
    AUFROC_0_100 = my_round_value(value=metrics['AUFROC']['[0, 100]'][-1], digits=3)
    AUPR = my_round_value(value=metrics['AUPR'][-1], digits=3)

    # metrics timer conversion
    metrics_time_test = timer(time_elapsed=metrics['time']['test'][-1])
    metrics_time_metrics = timer(time_elapsed=metrics['time']['metrics'][-1])

    # save metrics-test.csv
    with open(metrics_path, 'w') as file:
        # writer
        writer = csv.writer(file)

        # write header
        header = metrics_header(metrics_type='test')
        writer.writerow(header)

        # write row
        writer.writerow([AUC,
                         sensitivity_10_FPS,
                         sensitivity_max,
                         AUFROC_0_1,
                         AUFROC_0_10,
                         AUFROC_0_50,
                         AUFROC_0_100,
                         AUPR,
                         "{} h {} m {} s".format(metrics_time_test['hours'], metrics_time_test['minutes'], metrics_time_test['seconds']),
                         "{} h {} m {} s".format(metrics_time_metrics['hours'], metrics_time_metrics['minutes'], metrics_time_metrics['seconds'])])
