import argparse
import copy

from torch.utils.data import Dataset, ConcatDataset

from net.dataset.dataset_transforms_augmentation import dataset_transforms_augmentation


def dataset_augmentation(normalization: str,
                         parser: argparse.Namespace,
                         statistics_path: str,
                         dataset_train: Dataset,
                         debug: bool):
    """
    Apply dataset augmentation (only for dataset-train)

    :param normalization: normalization type
    :param parser: parser of parameters-parsing
    :param statistics_path: statistics path
    :param dataset_train: dataset-train
    :param debug: debug option
    :return: dataset-train augmented
    """

    # dataset augmentation transforms
    transforms_augmentation = dataset_transforms_augmentation(normalization=normalization,
                                                              parser=parser,
                                                              statistics_path=statistics_path,
                                                              debug=debug)

    # dataset-train HorizontalFLip augmentation
    dataset_train_HorizontalFlip = copy.deepcopy(dataset_train)
    dataset_train_HorizontalFlip.dataset.transforms = transforms_augmentation['HorizontalFlip']

    # dataset-train VerticalFLip augmentation
    dataset_train_VerticalFlip = copy.deepcopy(dataset_train)
    dataset_train_VerticalFlip.dataset.transforms = transforms_augmentation['VerticalFlip']

    # dataset-train HorizontalAndVerticalFLip augmentation
    dataset_train_HorizontalAndVerticalFlip = copy.deepcopy(dataset_train)
    dataset_train_HorizontalAndVerticalFlip.dataset.transforms = transforms_augmentation['HorizontalAndVerticalFlip']

    # concat dataset-train
    dataset_train_augmented = ConcatDataset([dataset_train,
                                             dataset_train_HorizontalFlip,
                                             dataset_train_VerticalFlip,
                                             dataset_train_HorizontalAndVerticalFlip])

    return dataset_train_augmented
