from net.metrics.utility.my_round_value import my_round_value
from net.metrics.utility.timer import timer


def show_metrics_train(metrics: dict,
                       work_point: int):
    """
    Show metrics-train

    :param metrics: metrics dictionary
    :param work_point: work point
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

    print("\n--------------"
          "\nMETRICS TRAIN:"
          "\n--------------"
          "\nEpoch: {} | Loss: {}".format(ticks, loss),
          "\nClassification Loss: {} | Regression Loss: {}".format(classification_loss, regression_loss))

    print("\n-------------------"
          "\nMETRICS VALIDATION:"
          "\n-------------------"
          "\nAUC: {}".format(AUC),
          "\nSensitivity (Work Point {} avg FP for scan): {}".format(work_point, sensitivity_work_point),
          "\nSensitivity (Max): {}".format(sensitivity_max),
          "\nAUFROC [0, 1]: {}".format(AUFROC_0_1),
          "\nAUFROC [0, 10]: {}".format(AUFROC_0_10),
          "\nAUFROC [0, 50]: {}".format(AUFROC_0_50),
          "\nAUFROC [0, 100]: {}".format(AUFROC_0_100))

    print("\n-----"
          "\nTIME:"
          "\n-----"
          "\nTRAIN: {} h {} m {} s".format(metrics_time_train['hours'], metrics_time_train['minutes'], metrics_time_train['seconds']),
          "\nVALIDATION: {} h {} m {} s".format(metrics_time_validation['hours'], metrics_time_validation['minutes'], metrics_time_validation['seconds']),
          "\nMETRICS: {} h {} m {} s".format(metrics_time_metrics['hours'], metrics_time_metrics['minutes'], metrics_time_metrics['seconds']))
