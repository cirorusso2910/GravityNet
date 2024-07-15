import argparse
import sys

from torchvision import transforms

from net.dataset.statistics.min_max_statistics import read_min_max_statistics
from net.dataset.statistics.standard_statistics import read_std_statistics
from net.dataset.transforms.Add3ChannelsImage import Add3ChannelsImage
from net.dataset.transforms.AnnotationPadding import AnnotationPadding
from net.dataset.transforms.MinMaxNormalization import MinMaxNormalization
from net.dataset.transforms.Rescale import Rescale
from net.dataset.transforms.StandardNormalization import StandardNormalization
from net.dataset.transforms.ToTensor import ToTensor
from net.dataset.transforms_augmentation.MyHorizontalAndVerticalFlip import MyHorizontalAndVerticalFlip
from net.dataset.transforms_augmentation.MyHorizontalFlip import MyHorizontalFlip
from net.dataset.transforms_augmentation.MyVerticalFlip import MyVerticalFlip
from net.utility.msg.msg_error import msg_error


def dataset_transforms_augmentation(normalization: str,
                                    parser: argparse.Namespace,
                                    statistics_path: str) -> dict:
    """
    Collect dataset transforms augmentation (only for dataset-train)

    :param normalization: normalization type
    :param parser: parser of parameters-parsing
    :param statistics_path: statistics path
    :return: train transforms augmentation dictionary
    """

    # ---- #
    # NONE #
    # ---- #
    if normalization == 'none':

        # HorizontalFlip augmentation transforms
        train_augmentation_HorizontalFlip_transforms = transforms.Compose([
            # DATA
            Rescale(rescale=parser.rescale,
                    num_channels=parser.num_channels),  # Rescale images and annotations
            # DATA AUGMENTATION
            MyHorizontalFlip(num_channels=parser.num_channels),  # My Horizontal Flip
            # DATA PREPARATION
            Add3ChannelsImage(num_channels=parser.num_channels),  # Add 3 Channels to image [C, H, W]
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
        ])

        # VerticalFlip augmentation transforms
        train_augmentation_VerticalFlip_transforms = transforms.Compose([
            # DATA
            Rescale(rescale=parser.rescale,
                    num_channels=parser.num_channels),  # Rescale images and annotations
            # DATA AUGMENTATION
            MyVerticalFlip(num_channels=parser.num_channels),  # My Vertical Flip
            # DATA PREPARATION
            Add3ChannelsImage(num_channels=parser.num_channels),  # Add 3 Channels to image [C, H, W]
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
        ])

        # HorizontalFlip and VerticalFlip augmentation transforms
        train_augmentation_HorizontalFlip_and_VerticalFlip_transforms = transforms.Compose([
            # DATA
            Rescale(rescale=parser.rescale,
                    num_channels=parser.num_channels),  # Rescale images and annotations
            # DATA AUGMENTATION
            MyHorizontalAndVerticalFlip(num_channels=parser.num_channels),  # My Horizontal and Vertical Flip
            # DATA PREPARATION
            Add3ChannelsImage(num_channels=parser.num_channels),  # Add 3 Channels to image [C, H, W]
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
        ])

    # ------- #
    # MIN-MAX #
    # ------- #
    elif normalization == 'min-max':

        # read min-max statistics
        min_max_statistics = read_min_max_statistics(statistics_path=statistics_path)

        # HorizontalFlip augmentation transforms
        train_augmentation_HorizontalFlip_transforms = transforms.Compose([
            # DATA
            Rescale(rescale=parser.rescale,
                    num_channels=parser.num_channels),  # Rescale images and annotations
            # DATA AUGMENTATION
            MyHorizontalFlip(num_channels=parser.num_channels),  # My Horizontal Flip
            # DATA PREPARATION
            Add3ChannelsImage(num_channels=parser.num_channels),  # Add 3 Channels to image [C, H, W]
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
            # MIN-MAX NORMALIZATION
            MinMaxNormalization(min=min_max_statistics['train']['min'], max=min_max_statistics['train']['max'])  # min-max normalization
        ])

        # VerticalFlip augmentation transforms
        train_augmentation_VerticalFlip_transforms = transforms.Compose([
            # DATA
            Rescale(rescale=parser.rescale,
                    num_channels=parser.num_channels),  # Rescale images and annotations
            # DATA AUGMENTATION
            MyVerticalFlip(num_channels=parser.num_channels),  # My Vertical Flip
            # DATA PREPARATION
            Add3ChannelsImage(num_channels=parser.num_channels),  # Add 3 Channels to image [C, H, W]
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
            # MIN-MAX NORMALIZATION
            MinMaxNormalization(min=min_max_statistics['train']['min'], max=min_max_statistics['train']['max'])  # min-max normalization
        ])

        # HorizontalFlip and VerticalFlip augmentation transforms
        train_augmentation_HorizontalFlip_and_VerticalFlip_transforms = transforms.Compose([
            # DATA
            Rescale(rescale=parser.rescale,
                    num_channels=parser.num_channels),  # Rescale images and annotations
            # DATA AUGMENTATION
            MyHorizontalAndVerticalFlip(num_channels=parser.num_channels),  # My Horizontal and Vertical Flip
            # DATA PREPARATION
            Add3ChannelsImage(num_channels=parser.num_channels),  # Add 3 Channels to image [C, H, W]
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
            # MIN-MAX NORMALIZATION
            MinMaxNormalization(min=min_max_statistics['train']['min'], max=min_max_statistics['train']['max'])  # min-max normalization
        ])

    # --- #
    # STD #
    # --- #
    elif normalization == 'std':

        # read std statistics
        std_statistics = read_std_statistics(statistics_path=statistics_path)

        # HorizontalFlip augmentation transforms
        train_augmentation_HorizontalFlip_transforms = transforms.Compose([
            # DATA
            Rescale(rescale=parser.rescale,
                    num_channels=parser.num_channels),  # Rescale images and annotations
            # DATA AUGMENTATION
            MyHorizontalFlip(num_channels=parser.num_channels),  # My Horizontal Flip
            # DATA PREPARATION
            Add3ChannelsImage(num_channels=parser.num_channels),  # Add 3 Channels to image [C, H, W]
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
            # STANDARD NORMALIZATION
            StandardNormalization(mean=std_statistics['train']['mean'], std=std_statistics['train']['std'])  # standard normalization
        ])

        # VerticalFlip augmentation transforms
        train_augmentation_VerticalFlip_transforms = transforms.Compose([
            # DATA
            Rescale(rescale=parser.rescale,
                    num_channels=parser.num_channels),  # Rescale images and annotations
            # DATA AUGMENTATION
            MyVerticalFlip(num_channels=parser.num_channels),  # My Vertical Flip
            # DATA PREPARATION
            Add3ChannelsImage(num_channels=parser.num_channels),  # Add 3 Channels to image [C, H, W]
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
            # STANDARD NORMALIZATION
            StandardNormalization(mean=std_statistics['train']['mean'], std=std_statistics['train']['std'])  # standard normalization
        ])

        # HorizontalFlip and VerticalFlip augmentation transforms
        train_augmentation_HorizontalFlip_and_VerticalFlip_transforms = transforms.Compose([
            # DATA
            Rescale(rescale=parser.rescale,
                    num_channels=parser.num_channels),  # Rescale images and annotations
            # DATA AUGMENTATION
            MyHorizontalAndVerticalFlip(num_channels=parser.num_channels),  # My Horizontal and Vertical Flip
            # DATA PREPARATION
            Add3ChannelsImage(num_channels=parser.num_channels),  # Add 3 Channels to image [C, H, W]
            AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
            ToTensor(),  # To Tensor
            # STANDARD NORMALIZATION
            StandardNormalization(mean=std_statistics['train']['mean'], std=std_statistics['train']['std'])  # standard normalization
        ])

    else:
        str_err = msg_error(file=__file__,
                            variable=normalization,
                            type_variable="normalization type",
                            choices="[none, min-max, std]")
        sys.exit(str_err)

    transforms_augmentation = {
        'HorizontalFlip': train_augmentation_HorizontalFlip_transforms,
        'VerticalFlip': train_augmentation_VerticalFlip_transforms,
        'HorizontalAndVerticalFlip': train_augmentation_HorizontalFlip_and_VerticalFlip_transforms
    }

    return transforms_augmentation
