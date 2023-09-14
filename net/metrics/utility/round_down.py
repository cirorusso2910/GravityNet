import math


def round_down(value, decimals=0):
    """ round down value """

    multiplier = 10 ** decimals

    return int(math.floor(value * multiplier) / multiplier)
