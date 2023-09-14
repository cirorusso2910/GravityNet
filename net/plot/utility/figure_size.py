from typing import Tuple


def figure_size(epochs: int) -> Tuple[int, int]:
    """
    Define figure size for plots according to num of epochs

    :param epochs: num of epochs
    :return: figure size
    """

    if epochs <= 10:
        figsize_x = 14
        figsize_y = 6
    elif 10 < epochs <= 30:
        figsize_x = 18
        figsize_y = 6
    elif 30 < epochs <= 50:
        figsize_x = 20
        figsize_y = 6
    elif 50 < epochs <= 80:
        figsize_x = 30
        figsize_y = 6
    elif 80 < epochs <= 100:
        figsize_x = 40
        figsize_y = 6
    else:
        figsize_x = 50
        figsize_y = 6

    return figsize_x, figsize_y
