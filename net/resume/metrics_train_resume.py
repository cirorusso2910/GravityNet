import csv

from net.initialization.header.metrics import metrics_header
from net.utility.msg.msg_metrics_complete import msg_metrics_complete
from net.metrics.utility.my_notation import scientific_notation


def metrics_train_resume_csv(metrics_path: str,
                             metrics: dict):
    """
    Resume metrics-train.csv

    :param metrics_path: metrics path
    :param metrics: metrics dictionary
    """

    with open(metrics_path, 'w') as file:
        writer = csv.writer(file)

        # write header
        header = metrics_header(metrics_type='train')
        writer.writerow(header)

        # iterate row writer
        for row in range(len(metrics['ticks'])):
            writer.writerow([metrics['ticks'][row],
                             metrics['loss']['loss'][row],
                             metrics['loss']['classification'][row],
                             metrics['loss']['regression'][row],
                             scientific_notation(number=metrics['learning_rate'][row]),
                             metrics['AUC'][row],
                             metrics['sensitivity']['work_point'][row],
                             metrics['sensitivity']['max'][row],
                             metrics['AUFROC']['[0, 1]'][row],
                             metrics['AUFROC']['[0, 10]'][row],
                             metrics['AUFROC']['[0, 50]'][row],
                             metrics['AUFROC']['[0, 100]'][row],
                             metrics['time']['train'][row],
                             metrics['time']['validation'][row],
                             metrics['time']['metrics'][row]])

    msg_metrics_complete(metrics_type='resume')
