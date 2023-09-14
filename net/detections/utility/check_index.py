from typing import List


def check_index(index_TP: List[int],
                index_positive: List[int]) -> List[int]:
    """
    Check index and delete elements in index_positive and index_TP:
    to avoid one prediction hooking into two different annotations

    :param index_TP: index of TP
    :param index_positive: index of positive predictions
    :return: index of positive predictions checked
    """

    index_positive = as_list(x=index_positive)

    item_to_delete = [item for item in index_positive if item in index_TP]

    for i in item_to_delete:
        index_positive.remove(i)

    return index_positive


def as_list(x):
    """
    Check if x is a list

    :param x: list to be checked
    """

    if type(x) is list:
        return x
    else:
        return [x]
