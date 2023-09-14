from torch.utils.data import Dataset, Subset
from typing import Tuple

from net.dataset.utility.split_index import split_index


def dataset_split(data_split: dict,
                  dataset: Dataset) -> Tuple[Subset, Subset, Subset]:
    """
    Dataset split

    :param data_split: data split dictionary
    :param dataset: dataset
    :return: dataset train,
             dataset validation,
             dataset test
    """

    # split index
    train_index, validation_index, test_index = split_index(data_split=data_split)

    # subset dataset-train
    dataset_train = Subset(dataset=dataset,
                           indices=train_index)

    # subset dataset-val
    dataset_val = Subset(dataset=dataset,
                         indices=validation_index)

    # subset dataset-test
    dataset_test = Subset(dataset=dataset,
                          indices=test_index)

    return dataset_train, dataset_val, dataset_test
