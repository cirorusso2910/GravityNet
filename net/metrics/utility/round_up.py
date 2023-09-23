import math


def round_up(value, decimals=0):
    """
    Round up value

    :param value: value
    :param decimals: decimals
    :return: value rounded up
    """

    multiplier = 10 ** decimals

    return math.ceil(value * multiplier) / multiplier
