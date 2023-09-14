import math


def round_up(value, decimals=0):
    """ round up value """

    multiplier = 10 ** decimals

    return math.ceil(value * multiplier) / multiplier
