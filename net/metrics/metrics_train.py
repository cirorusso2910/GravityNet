import csv
import os

from net.initialization.header.metrics import metrics_header
from net.metrics.utility.my_round_value import my_round_value
from net.metrics.utility.timer import timer


def metrics_train_csv(metrics_path: str,
                      metrics: dict):
    """
    Save metrics-train.csv

    :param metrics_path: metrics path
    :param metrics: metrics dictionary
    """

    # metrics round
    ticks = metrics['ticks'][-1]
    loss = my_round_value(value=metrics['loss']['loss'][-1], digits=3)
    classification_loss = my_round_value(value=metrics['loss']['classification'][-1], digits=3)
    regression_loss = my_round_value(value=metrics['loss']['regression'][-1], digits=3)
    AUC = my_round_value(value=metrics['AUC'][-1], digits=3)
    learning_rate = metrics['learning_rate'][-1]
    sensitivity_work_point = my_round_value(value=metrics['sensitivity']['work_point'][-1], digits=3)
    sensitivity_max = my_round_value(value=metrics['sensitivity']['max'][-1], digits=3)
    AUFROC_0_1 = my_round_value(value=metrics['AUFROC']['[0, 1]'][-1], digits=3)
    AUFROC_0_10 = my_round_value(value=metrics['AUFROC']['[0, 10]'][-1], digits=3)
    AUFROC_0_50 = my_round_value(value=metrics['AUFROC']['[0, 50]'][-1], digits=3)
    AUFROC_0_100 = my_round_value(value=metrics['AUFROC']['[0, 100]'][-1], digits=3)

    # metrics timer conversion
    metrics_time_train = timer(time_elapsed=metrics['time']['train'][-1])
    metrics_time_validation = timer(time_elapsed=metrics['time']['validation'][-1])
    metrics_time_metrics = timer(time_elapsed=metrics['time']['metrics'][-1])

    # check if file exists
    file_exists = os.path.isfile(metrics_path)

    # save metrics-train.csv
    with open(metrics_path, 'a') as file:
        # writer
        writer = csv.writer(file)

        if not file_exists:
            # write header
            header = metrics_header(metrics_type='train')
            writer.writerow(header)

        # write row
        writer.writerow([ticks,
                         loss,
                         classification_loss,
                         regression_loss,
                         learning_rate,
                         AUC,
                         sensitivity_work_point,
                         sensitivity_max,
                         AUFROC_0_1,
                         AUFROC_0_10,
                         AUFROC_0_50,
                         AUFROC_0_100,
                         "{} h {} m {} s".format(metrics_time_train['hours'], metrics_time_train['minutes'], metrics_time_train['seconds']),
                         "{} h {} m {} s".format(metrics_time_validation['hours'], metrics_time_validation['minutes'], metrics_time_validation['seconds']),
                         "{} h {} m {} s".format(metrics_time_metrics['hours'], metrics_time_metrics['minutes'], metrics_time_metrics['seconds'])])

