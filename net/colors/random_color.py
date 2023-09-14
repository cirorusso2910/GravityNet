import random

from typing import List


def random_colors() -> List:
    """
    Define a random colors [b,g,r]

    :return: random colors [b,g,r]
    """

    b = random.randint(0, 255)  # random blue channel
    g = random.randint(0, 255)  # random green channel
    r = random.randint(0, 255)  # random red channel

    bgr = [b, g, r]  # BGR list

    return bgr
