import math
from typing import List

def conversion_item_list_distance(item: List):
    """
    Converts single detections item with distance evaluation:
        - filename
        - num prediction
        - label
        - prediction x
        - prediction y
        - annotation x
        - annotation y

    :param item: detection item
    """

    item[1] = int(item[1])  # num prediction
    item[2] = int(item[2])  # label
    item[4] = int(item[4])  # prediction x
    item[5] = int(item[5])  # prediction y
    if not math.isnan(item[6]):
        item[6] = int(item[6])  # annotation x
    if not math.isnan(item[7]):
        item[7] = int(item[7])  # annotation y


def conversion_item_list_radius(item: List):
    """
    Converts single detections item with radius evaluation:
        - filename
        - num prediction
        - label
        - prediction x
        - prediction y
        - annotation x
        - annotation y
        - radius annotation

    :param item: detection item
    """

    item[1] = int(item[1])  # num prediction
    item[2] = int(item[2])  # label
    item[4] = int(item[4])  # prediction x
    item[5] = int(item[5])  # prediction y
    if not math.isnan(item[6]):
        item[6] = int(item[6])  # annotation x
    if not math.isnan(item[7]):
        item[7] = int(item[7])  # annotation y
    if not math.isnan(item[8]):
        item[8] = int(item[8])  # radius annotation
