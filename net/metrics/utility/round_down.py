import math


def round_down(value, decimals=0):
    """
    Round down value

    :param value: value
    :param decimals: decimals
    :return: value rounded down
    """

    multiplier = 10 ** decimals

    return int(math.floor(value * multiplier) / multiplier)
