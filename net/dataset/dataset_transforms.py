import argparse
import sys

from torchvision import transforms
from torchvision.transforms import Compose
from typing import Tuple

from net.dataset.statistics.min_max_statistics import read_min_max_statistics
from net.dataset.statistics.standard_statistics import read_std_statistics
from net.dataset.transforms.AnnotationPadding import AnnotationPadding
from net.dataset.transforms.MinMaxNormalization import MinMaxNormalization
from net.dataset.transforms.Rescale import Rescale
from net.dataset.transforms.StandardNormalization import StandardNormalization
from net.dataset.transforms.ToTensor import ToTensor
from net.utility.msg.msg_error import msg_error


def dataset_transforms(normalization: str,
                       parser: argparse.Namespace,
                       statistics_path: str,
                       debug: bool) -> Tuple[Compose, Compose, Compose]:
    """
    Collect dataset transforms

    :param normalization: normalization type
    :param parser: parser of parameters-parsing
    :param statistics_path: statistics path
    :param debug: debug option (not used)
    :return: train transforms,
             validation transforms,
             test transforms
    """

    # ---- #
    # NONE #
    # ---- #
    if normalization == 'none':

        # train dataset transforms
        train_transforms = transforms.Compose([
            # PRE-PROCESSING or DATA
            # $TRANSFORMS_PRE_PROCESSING$ or $TRANSFORMS_DATA$
            Rescale(rescale=parser.rescale),
            # DATA PREPARATION
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
        ])

        # validation dataset transforms
        val_transforms = transforms.Compose([
            # PRE-PROCESSING or DATA
            # $TRANSFORMS_PRE_PROCESSING$ or $TRANSFORMS_DATA$
            Rescale(rescale=parser.rescale),
            # DATA PREPARATION
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
        ])

        # test dataset transforms
        test_transforms = transforms.Compose([
            # PRE-PROCESSING or DATA
            # $TRANSFORMS_PRE_PROCESSING$ or $TRANSFORMS_DATA$
            Rescale(rescale=parser.rescale),
            # DATA PREPARATION
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
        ])

    # ------- #
    # MIN-MAX #
    # ------- #
    elif normalization == 'min-max':

        # read min-max statistics
        min_max_statistics = read_min_max_statistics(statistics_path=statistics_path)

        # train dataset transforms
        train_transforms = transforms.Compose([
            # PRE-PROCESSING or DATA
            # $TRANSFORMS_PRE_PROCESSING$ or $TRANSFORMS_DATA$
            Rescale(rescale=parser.rescale),
            # DATA PREPARATION
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
            # MIN-MAX NORMALIZATION
            MinMaxNormalization(min=min_max_statistics['train']['min'], max=min_max_statistics['train']['max'])  # min-max normalization
        ])

        # validation dataset transforms
        val_transforms = transforms.Compose([
            # PRE-PROCESSING or DATA
            # $TRANSFORMS_PRE_PROCESSING$ or $TRANSFORMS_DATA$
            Rescale(rescale=parser.rescale),
            # DATA PREPARATION
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
            # MIN-MAX NORMALIZATION
            MinMaxNormalization(min=min_max_statistics['validation']['min'], max=min_max_statistics['validation']['max'])  # min-max normalization
        ])

        # test dataset transforms
        test_transforms = transforms.Compose([
            # PRE-PROCESSING or DATA
            # $TRANSFORMS_PRE_PROCESSING$ or $TRANSFORMS_DATA$
            Rescale(rescale=parser.rescale),
            # DATA PREPARATION
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
            # MIN-MAX NORMALIZATION
            MinMaxNormalization(min=min_max_statistics['test']['min'], max=min_max_statistics['test']['max'])  # min-max normalization
        ])

    # --- #
    # STD #
    # --- #
    elif normalization == 'std':

        # read std statistics
        std_statistics = read_std_statistics(statistics_path=statistics_path)

        # train dataset transforms
        train_transforms = transforms.Compose([
            # PRE-PROCESSING or DATA
            # $TRANSFORMS_PRE_PROCESSING$ or $TRANSFORMS_DATA$
            Rescale(rescale=parser.rescale),
            # DATA PREPARATION
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
            # STANDARD NORMALIZATION
            StandardNormalization(mean=std_statistics['train']['mean'], std=std_statistics['train']['std'])  # standard normalization
        ])

        # validation dataset transforms
        val_transforms = transforms.Compose([
            # PRE-PROCESSING or DATA
            # $TRANSFORMS_PRE_PROCESSING$ or $TRANSFORMS_DATA$
            Rescale(rescale=parser.rescale),
            # DATA PREPARATION
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
            # STANDARD NORMALIZATION
            StandardNormalization(mean=std_statistics['validation']['mean'], std=std_statistics['validation']['std'])  # standard normalization
        ])

        # test dataset transforms
        test_transforms = transforms.Compose([
            # PRE-PROCESSING or DATA
            # $TRANSFORMS_PRE_PROCESSING$ or $TRANSFORMS_DATA$
            Rescale(rescale=parser.rescale),
            # DATA PREPARATION
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
            # STANDARD NORMALIZATION
            StandardNormalization(mean=std_statistics['test']['mean'], std=std_statistics['test']['std'])  # standard normalization
        ])

    else:
        str_err = msg_error(file=__file__,
                            variable=normalization,
                            type_variable="normalization type",
                            choices="[none, min-max, std]")
        sys.exit(str_err)

    return train_transforms, val_transforms, test_transforms
