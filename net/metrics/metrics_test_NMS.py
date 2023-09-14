import csv

from net.initialization.header.metrics import metrics_header
from net.metrics.utility.my_round_value import my_round_value
from net.metrics.utility.timer import timer


def metrics_test_NMS_csv(metrics_path: str,
                         metrics: dict):
    """
    Save metrics-test-NMS.csv

    :param metrics_path: metrics path
    :param metrics: metrics dictionary
    """

    # metrics round
    AUC = my_round_value(value=metrics['AUC'][-1], digits=3)
    sensitivity_work_point = my_round_value(value=metrics['sensitivity']['work_point'][-1], digits=3)
    sensitivity_max = my_round_value(value=metrics['sensitivity']['max'][-1], digits=3)
    AUFROC_0_1 = my_round_value(value=metrics['AUFROC']['[0, 1]'][-1], digits=3)
    AUFROC_0_10 = my_round_value(value=metrics['AUFROC']['[0, 10]'][-1], digits=3)
    AUFROC_0_50 = my_round_value(value=metrics['AUFROC']['[0, 50]'][-1], digits=3)
    AUFROC_0_100 = my_round_value(value=metrics['AUFROC']['[0, 100]'][-1], digits=3)

    # metrics timer conversion
    metrics_time_test_NMS = timer(time_elapsed=metrics['time']['NMS'][-1])
    metrics_time_metrics = timer(time_elapsed=metrics['time']['metrics'][-1])

    with open(metrics_path, 'w') as file:
        # writer
        writer = csv.writer(file)

        # write header
        header = metrics_header(metrics_type='test_NMS')
        writer.writerow(header)

        # write row
        writer.writerow([AUC,
                         sensitivity_work_point,
                         sensitivity_max,
                         AUFROC_0_1,
                         AUFROC_0_10,
                         AUFROC_0_50,
                         AUFROC_0_100,
                         "{} h {} m {} s".format(metrics_time_test_NMS['hours'], metrics_time_test_NMS['minutes'], metrics_time_test_NMS['seconds']),
                         "{} h {} m {} s".format(metrics_time_metrics['hours'], metrics_time_metrics['minutes'], metrics_time_metrics['seconds'])])
