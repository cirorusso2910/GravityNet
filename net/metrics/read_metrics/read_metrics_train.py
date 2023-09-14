from pandas import read_csv

from net.initialization.header.metrics import metrics_header


def read_metrics_train_csv(metrics_path: str) -> dict:
    """
    Read metrics-train.csv

    :param metrics_path: metrics path
    :return: metrics dictionary
    """

    # read metrics
    header = metrics_header(metrics_type='train')
    metrics_train = read_csv(filepath_or_buffer=metrics_path, usecols=header)
    print("metrics-train.csv reading: COMPLETE")

    # metrics list
    ticks = metrics_train['EPOCH'].tolist()
    loss = metrics_train['LOSS'].tolist()
    classification_loss = metrics_train['CLASSIFICATION LOSS'].tolist()
    regression_loss = metrics_train['REGRESSION LOSS'].tolist()
    learning_rate = metrics_train['LEARNING RATE'].tolist()
    AUC = metrics_train['AUC'].tolist()
    sensitivity_work_point = metrics_train['SENSITIVITY WORK POINT'].tolist()
    sensitivity_max = metrics_train['SENSITIVITY MAX'].tolist()
    AUFROC_0_1 = metrics_train['AUFROC [0, 1]'].tolist()
    AUFROC_0_10 = metrics_train['AUFROC [0, 10]'].tolist()
    AUFROC_0_50 = metrics_train['AUFROC [0, 50]'].tolist()
    AUFROC_0_100 = metrics_train['AUFROC [0, 100]'].tolist()
    time_train = metrics_train['TIME TRAIN'].tolist()
    time_validation = metrics_train['TIME VALIDATION'].tolist()
    time_metrics = metrics_train['TIME METRICS'].tolist()

    # metrics train dict
    metrics = {
        'ticks': ticks,
        'loss': {
            'loss': loss,
            'classification': classification_loss,
            'regression': regression_loss
        },

        'learning_rate': learning_rate,
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
        },

        'time': {
            'train': time_train,
            'validation': time_validation,
            'metrics': time_metrics
        }
    }

    return metrics
