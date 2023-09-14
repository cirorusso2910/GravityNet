from pandas import read_csv

from net.initialization.header.metrics import metrics_header


def read_metrics_sensitivity_images_csv(metrics_path: str) -> dict:
    """
    Read metrics-sensitivity-image.csv

    :param metrics_path: metrics path
    :return metrics dictionary
    """

    # read metrics
    header = metrics_header(metrics_type='sensitivity')
    metrics_sensitivity_images = read_csv(filepath_or_buffer=metrics_path, usecols=header)
    print("metrics-sensitivity-images.csv reading: COMPLETE")

    # metrics list
    image = metrics_sensitivity_images['IMAGE'].tolist()
    annotations = metrics_sensitivity_images['ANNOTATIONS'].tolist()
    TP = metrics_sensitivity_images['TP'].tolist()
    FN = metrics_sensitivity_images['FN'].tolist()
    FP = metrics_sensitivity_images['FP'].tolist()
    sensitivity = metrics_sensitivity_images['SENSITIVITY'].tolist()

    # performance metrics test
    metrics = {
        'IMAGE': image,
        'ANNOTATIONS': annotations,
        'TP': TP,
        'FN': FN,
        'FP': FP,
        'SENSITIVITY': sensitivity
    }

    return metrics
