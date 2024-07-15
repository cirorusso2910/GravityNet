from net.metrics.utility.my_round_value import my_round_value
from net.metrics.utility.timer import timer


def show_metrics_test(metrics: dict):
    """
    Show metrics-test

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

    print("\n-------------"
          "\nMETRICS TEST:"
          "\n-------------"
          "\nAUC: {}".format(AUC),
          "\nSensitivity 10 FPS: {}".format(sensitivity_10_FPS),
          "\nSensitivity Max: {}".format(sensitivity_max),
          "\nAUFROC [0, 1]: {}".format(AUFROC_0_1),
          "\nAUFROC [0, 10]: {}".format(AUFROC_0_10),
          "\nAUFROC [0, 50]: {}".format(AUFROC_0_50),
          "\nAUFROC [0, 100]: {}".format(AUFROC_0_100),
          "\nAUPR: {}".format(AUPR))

    print("\n-----"
          "\nTIME:"
          "\n-----"
          "\nTEST: {} h {} m {} s".format(metrics_time_test['hours'], metrics_time_test['minutes'], metrics_time_test['seconds']),
          "\nMETRICS: {} h {} m {} s".format(metrics_time_metrics['hours'], metrics_time_metrics['minutes'], metrics_time_metrics['seconds']))
