from typing import List, Tuple


def best_metrics(metric: List) -> Tuple[float, int]:
    """
    Get best metrics value and index

    :param metric: metric
    :return: metrics value,
             index
    """

    # max metric
    max_metric = max(metric)

    # index best metric
    index_best_metric = metric.index(max_metric) + 1  # + 1 (epochs start from 1)

    return max_metric, index_best_metric
