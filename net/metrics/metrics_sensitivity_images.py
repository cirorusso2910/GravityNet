import csv
import os.path

from typing import List

from net.initialization.header.metrics import metrics_header


def metrics_sensitivity_images(metrics_path: str,
                               row: List):
    """
    Save metrics-sensitivity-images.csv

    :param metrics_path: metrics path
    :param row: row to write
    """

    # check if file exists
    file_exists = os.path.isfile(metrics_path)

    with open(metrics_path, 'a') as f:
        writer = csv.writer(f)

        # write header
        if not file_exists:
            header = metrics_header(metrics_type='sensitivity')
            writer.writerow(header)

        writer.writerows([row])
