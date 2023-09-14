import argparse
import sys

from torchvision import transforms
from torchvision.transforms import Compose
from typing import Tuple

from net.dataset.statistics.min_max_statistics import read_min_max_statistics
from net.dataset.statistics.standard_statistics import read_std_statistics
from net.dataset.transforms.EophthaMA.AnnotationPadding import AnnotationPadding as AnnotationPadding_EophthaMA
from net.dataset.transforms.EophthaMA.MinMaxNormalization import MinMaxNormalization as MinMaxNormalization_EophthaMA
from net.dataset.transforms.EophthaMA.Rescale import Rescale as Rescale_EophthaMA
from net.dataset.transforms.EophthaMA.Resize import Resize as Resize_EophthaMA
from net.dataset.transforms.EophthaMA.SelectImageChannel import SelectImageChannel as SelectImageChannel_EophthaMA
from net.dataset.transforms.EophthaMA.SelectMaxRadius import SelectMaxRadius as SelectMaxRadius_EophthaMA
from net.dataset.transforms.EophthaMA.StandardNormalization import StandardNormalization as StandardNormalization_EophthaMA
from net.dataset.transforms.EophthaMA.ToTensor import ToTensor as ToTensor_EophthaMA
from net.dataset.transforms.INbreast.Add3ChannelsImage import Add3ChannelsImage as Add3ChannelsImage_INbreast
from net.dataset.transforms.INbreast.AnnotationCheck import AnnotationCheck as AnnotationCheck_INbreast
from net.dataset.transforms.INbreast.Crop import Crop as Crop_INbreast
from net.dataset.transforms.INbreast.Flip import Flip as Flip_INbreast
from net.dataset.transforms.INbreast.MinMaxNormalization import MinMaxNormalization as MinMaxNormalization_INbreast
from net.dataset.transforms.INbreast.RadiusFilter import RadiusFilter as RadiusFilter_INbreast
from net.dataset.transforms.INbreast.AnnotationPadding import AnnotationPadding as AnnotationPadding_INbreast
from net.dataset.transforms.INbreast.Annotation import Annotation as Annotation_INbreast
from net.dataset.transforms.INbreast.Rescale import Rescale as Rescale_INbreast
from net.dataset.transforms.INbreast.SelectROI import SelectROI as SelectROI_INbreast
from net.dataset.transforms.INbreast.StandardNormalization import StandardNormalization as StandardNormalization_INbreast
from net.dataset.transforms.INbreast.ToTensor import ToTensor as ToTensor_INbreast
from net.utility.msg.msg_error import msg_error


def dataset_transforms(dataset: str,
                       normalization: str,
                       parser: argparse.Namespace,
                       statistics_path: str,
                       debug: bool) -> Tuple[Compose, Compose, Compose]:
    """
    Collect dataset transforms

    :param dataset: dataset name
    :param normalization: normalization type
    :param parser: parser of parameters-parsing
    :param statistics_path: statistics path
    :param debug: debug option (not used)
    :return: train transforms,
             validation transforms,
             test transforms
    """

    # ----------- #
    # E-ophtha-MA #
    # ----------- #
    if dataset == 'E-ophtha-MA':

        # ---- #
        # NONE #
        # ---- #
        if normalization == 'none':

            # train dataset transforms
            train_transforms = transforms.Compose([
                # DATA
                Resize_EophthaMA(size=(parser.image_height_resize, parser.image_width_resize),
                                 tool=parser.resize_tool),  # Resize image and annotations
                Rescale_EophthaMA(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                SelectImageChannel_EophthaMA(channel=parser.channel),  # select image channel [RGB, G]
                SelectMaxRadius_EophthaMA(),  # select radius
                AnnotationPadding_EophthaMA(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
                ToTensor_EophthaMA(),  # To Tensor
            ])

            # validation dataset transforms
            val_transforms = transforms.Compose([
                # DATA
                Resize_EophthaMA(size=(parser.image_height_resize, parser.image_width_resize),
                                 tool=parser.resize_tool),  # Resize image and annotations
                Rescale_EophthaMA(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                SelectImageChannel_EophthaMA(channel=parser.channel),  # select image channel [RGB, G]
                SelectMaxRadius_EophthaMA(),  # select radius
                AnnotationPadding_EophthaMA(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
                ToTensor_EophthaMA(),  # To Tensor
            ])

            # test dataset transforms
            test_transforms = transforms.Compose([
                # DATA
                Resize_EophthaMA(size=(parser.image_height_resize, parser.image_width_resize),
                                 tool=parser.resize_tool),  # Resize image and annotations
                Rescale_EophthaMA(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                SelectImageChannel_EophthaMA(channel=parser.channel),  # select image channel [RGB, G]
                SelectMaxRadius_EophthaMA(),  # select radius
                AnnotationPadding_EophthaMA(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
                ToTensor_EophthaMA(),  # To Tensor
            ])

        # ------- #
        # MIN-MAX #
        # ------- #
        elif normalization == 'min-max':

            # read min-max statistics
            min_max_statistics = read_min_max_statistics(statistics_path=statistics_path)

            # train dataset transforms
            train_transforms = transforms.Compose([
                # DATA
                Resize_EophthaMA(size=(parser.image_height_resize, parser.image_width_resize),
                                 tool=parser.resize_tool),  # Resize image and annotations
                Rescale_EophthaMA(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                SelectImageChannel_EophthaMA(channel=parser.channel),  # select image channel [RGB, G]
                SelectMaxRadius_EophthaMA(),  # select radius
                AnnotationPadding_EophthaMA(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
                ToTensor_EophthaMA(),  # To Tensor
                # MIN-MAX NORMALIZATION
                MinMaxNormalization_EophthaMA(min=min_max_statistics['train']['min'], max=min_max_statistics['train']['max'])  # min-max normalization
            ])

            # validation dataset transforms
            val_transforms = transforms.Compose([
                # DATA
                Resize_EophthaMA(size=(parser.image_height_resize, parser.image_width_resize),
                                 tool=parser.resize_tool),  # Resize image and annotations
                Rescale_EophthaMA(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                SelectImageChannel_EophthaMA(channel=parser.channel),  # select image channel [RGB, G]
                SelectMaxRadius_EophthaMA(),  # select radius
                AnnotationPadding_EophthaMA(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
                ToTensor_EophthaMA(),  # To Tensor
                # MIN-MAX NORMALIZATION
                MinMaxNormalization_EophthaMA(min=min_max_statistics['validation']['min'], max=min_max_statistics['validation']['max'])  # min-max normalization
            ])

            # test dataset transforms
            test_transforms = transforms.Compose([
                # DATA
                Resize_EophthaMA(size=(parser.image_height_resize, parser.image_width_resize),
                                 tool=parser.resize_tool),  # Resize image and annotations
                Rescale_EophthaMA(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                SelectImageChannel_EophthaMA(channel=parser.channel),  # select image channel [RGB, G]
                SelectMaxRadius_EophthaMA(),  # select radius
                AnnotationPadding_EophthaMA(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
                ToTensor_EophthaMA(),  # To Tensor
                # MIN-MAX NORMALIZATION
                MinMaxNormalization_EophthaMA(min=min_max_statistics['test']['min'], max=min_max_statistics['test']['max'])  # min-max normalization
            ])

        # --- #
        # STD #
        # --- #
        elif normalization == 'std':

            # read std statistics
            std_statistics = read_std_statistics(statistics_path=statistics_path)

            # train dataset transforms
            train_transforms = transforms.Compose([
                # DATA
                Resize_EophthaMA(size=(parser.image_height_resize, parser.image_width_resize),
                                 tool=parser.resize_tool),  # Resize image and annotations
                Rescale_EophthaMA(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                SelectImageChannel_EophthaMA(channel=parser.channel),  # select image channel [RGB, G]
                SelectMaxRadius_EophthaMA(),  # select radius
                AnnotationPadding_EophthaMA(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
                ToTensor_EophthaMA(),  # To Tensor
                # STANDARD NORMALIZATION
                StandardNormalization_EophthaMA(mean=std_statistics['train']['mean'], std=std_statistics['train']['std'])  # standard normalization
            ])

            # validation dataset transforms
            val_transforms = transforms.Compose([
                # DATA
                Resize_EophthaMA(size=(parser.image_height_resize, parser.image_width_resize),
                                 tool=parser.resize_tool),  # Resize image and annotations
                Rescale_EophthaMA(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                SelectImageChannel_EophthaMA(channel=parser.channel),  # select image channel [RGB, G]
                SelectMaxRadius_EophthaMA(),  # select radius
                AnnotationPadding_EophthaMA(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
                ToTensor_EophthaMA(),  # To Tensor
                # STANDARD NORMALIZATION
                StandardNormalization_EophthaMA(mean=std_statistics['validation']['mean'], std=std_statistics['validation']['std'])  # standard normalization
            ])

            # test dataset transforms
            test_transforms = transforms.Compose([
                # DATA
                Resize_EophthaMA(size=(parser.image_height_resize, parser.image_width_resize),
                                 tool=parser.resize_tool),  # Resize image and annotations
                Rescale_EophthaMA(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                SelectImageChannel_EophthaMA(channel=parser.channel),  # select image channel [RGB, G]
                SelectMaxRadius_EophthaMA(),  # select radius
                AnnotationPadding_EophthaMA(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
                ToTensor_EophthaMA(),  # To Tensor
                # STANDARD NORMALIZATION
                StandardNormalization_EophthaMA(mean=std_statistics['test']['mean'], std=std_statistics['test']['std'])  # standard normalization
            ])

        else:
            str_err = msg_error(file=__file__,
                                variable=normalization,
                                type_variable="normalization type",
                                choices="[none, min-max, std]")
            sys.exit(str_err)

    # -------- #
    # INbreast #
    # -------- #
    elif dataset == 'INbreast':

        # ---- #
        # NONE #
        # ---- #
        if normalization == 'none':

            # train dataset transforms
            train_transforms = transforms.Compose([
                # PRE PROCESSING
                SelectROI_INbreast(roi_type='Calcification'),  # select ROIs (annotations) type 'Calcification'
                AnnotationCheck_INbreast(),  # Annotation shape check (for empty)
                Flip_INbreast(orientation=parser.orientation),  # Flip images and annotations on orientation side
                Crop_INbreast(image_crop_shape=(parser.image_height_crop, parser.image_width_crop),
                              orientation=parser.orientation),  # Crop images and annotations (same shape)
                # DATA
                Rescale_INbreast(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                Annotation_INbreast(),  # Annotation definition
                Add3ChannelsImage_INbreast(),  # Add 3 Channels to image [C, H, W]
                AnnotationPadding_INbreast(max_padding=parser.max_padding),  # Annotation padding (for batch dataloader)
                ToTensor_INbreast()  # To Tensor
            ])

            # validation dataset transforms
            val_transforms = transforms.Compose([
                # PRE-PROCESSING
                SelectROI_INbreast(roi_type='Calcification'),  # select ROIs (annotations) type 'Calcification'
                AnnotationCheck_INbreast(),  # Annotation shape check (for empty)
                Flip_INbreast(orientation=parser.orientation),  # Flip all images and annotations on orientation side
                Crop_INbreast(image_crop_shape=(parser.image_height_crop, parser.image_width_crop),
                              orientation=parser.orientation),  # Crop all images and ROIs to have same shape
                # DATA EVALUATION (only in inference)
                RadiusFilter_INbreast(radius=7),  # ignore calcifications with radius >= 7 pixel
                # DATA
                Rescale_INbreast(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                Annotation_INbreast(),  # Annotation To Numpy
                Add3ChannelsImage_INbreast(),  # Add 3 Channels to image [C, H, W]
                AnnotationPadding_INbreast(max_padding=parser.max_padding),  # Annotation padding (for batch dataloader)
                ToTensor_INbreast()  # To Tensor
            ])

            # test dataset transforms
            test_transforms = transforms.Compose([
                # PRE-PROCESSING
                SelectROI_INbreast(roi_type='Calcification'),  # select ROIs (annotations) type 'Calcification'
                AnnotationCheck_INbreast(),  # Annotation shape check (for empty)
                Flip_INbreast(orientation=parser.orientation),  # Flip all images and annotations on orientation side
                Crop_INbreast(image_crop_shape=(parser.image_height_crop, parser.image_width_crop),
                              orientation=parser.orientation),  # Crop all images and ROIs to have same shape
                # DATA EVALUATION (only in inference)
                RadiusFilter_INbreast(radius=7),  # ignore calcifications with radius >= 7 pixel
                # DATA
                Rescale_INbreast(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                Annotation_INbreast(),  # Annotation definition
                Add3ChannelsImage_INbreast(),  # Add 3 Channels to image [C, H, W]
                AnnotationPadding_INbreast(max_padding=parser.max_padding),  # Annotation padding (for batch dataloader)
                ToTensor_INbreast()  # To Tensor
            ])

        # ------- #
        # MIN-MAX #
        # ------- #
        elif normalization == 'min-max':

            # read min-max statistics
            min_max_statistics = read_min_max_statistics(statistics_path=statistics_path)

            # train dataset transforms
            train_transforms = transforms.Compose([
                # PRE-PROCESSING
                SelectROI_INbreast(roi_type='Calcification'),  # select ROIs (annotations) type 'Calcification'
                AnnotationCheck_INbreast(),  # Annotation shape check (for empty)
                Flip_INbreast(orientation=parser.orientation),  # Flip images and annotations on orientation side
                Crop_INbreast(image_crop_shape=(parser.image_height_crop, parser.image_width_crop),
                              orientation=parser.orientation),  # Crop images and ROIs (same shape)
                # DATA
                Rescale_INbreast(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                Annotation_INbreast(),  # Annotation definition
                Add3ChannelsImage_INbreast(),  # Add 3 Channels to image [C, H, W]
                AnnotationPadding_INbreast(max_padding=parser.max_padding),  # Annotation padding (for batch dataloader)
                ToTensor_INbreast(),  # To Tensor
                # MIN-MAX NORMALIZATION
                MinMaxNormalization_INbreast(min=min_max_statistics['train']['min'], max=min_max_statistics['train']['max'])  # min-max normalization
            ])

            # validation dataset transforms
            val_transforms = transforms.Compose([
                # PRE-PROCESSING
                SelectROI_INbreast(roi_type='Calcification'),  # select ROIs (annotations) type 'Calcification'
                AnnotationCheck_INbreast(),  # Annotation shape check (for empty)
                Flip_INbreast(orientation=parser.orientation),  # Flip all images and annotations on orientation side
                Crop_INbreast(image_crop_shape=(parser.image_height_crop, parser.image_width_crop),
                              orientation=parser.orientation),  # Crop all images and ROIs to have same shape
                # DATA EVALUATION (only in inference)
                RadiusFilter_INbreast(radius=7),  # ignore calcifications with radius >= 7 pixel
                # DATA
                Rescale_INbreast(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                Annotation_INbreast(),  # Annotation definition
                Add3ChannelsImage_INbreast(),  # Add 3 Channels to image [C, H, W]
                AnnotationPadding_INbreast(max_padding=parser.max_padding),  # Annotation padding (for batch dataloader)
                ToTensor_INbreast(),  # To Tensor
                # MIN-MAX NORMALIZATION
                MinMaxNormalization_INbreast(min=min_max_statistics['validation']['min'], max=min_max_statistics['validation']['max'])  # min-max normalization
            ])

            # test dataset transforms
            test_transforms = transforms.Compose([
                # PRE-PROCESSING
                SelectROI_INbreast(roi_type='Calcification'),  # select ROIs (annotations) type 'Calcification'
                AnnotationCheck_INbreast(),  # Annotation shape check (for empty)
                Flip_INbreast(orientation=parser.orientation),  # Flip all images and annotations on orientation side
                Crop_INbreast(image_crop_shape=(parser.image_height_crop, parser.image_width_crop),
                              orientation=parser.orientation),  # Crop all images and ROIs to have same shape
                # DATA EVALUATION (only in inference)
                RadiusFilter_INbreast(radius=7),  # ignore calcifications with radius >= 7 pixel
                # DATA
                Rescale_INbreast(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                Annotation_INbreast(),  # Annotation definition
                Add3ChannelsImage_INbreast(),  # Add 3 Channels to image [C, H, W]
                AnnotationPadding_INbreast(max_padding=parser.max_padding),  # Annotation padding (for batch dataloader)
                ToTensor_INbreast(),  # To Tensor
                # MIN-MAX NORMALIZATION
                MinMaxNormalization_INbreast(min=min_max_statistics['test']['min'], max=min_max_statistics['test']['max'])  # min-max normalization
            ])

        # --- #
        # STD #
        # --- #
        elif normalization == 'std':

            # read std statistics
            std_statistics = read_std_statistics(statistics_path=statistics_path)

            # train dataset transforms
            train_transforms = transforms.Compose([
                # PRE-PROCESSING
                SelectROI_INbreast(roi_type='Calcification'),  # select ROIs (annotations) type 'Calcification'
                AnnotationCheck_INbreast(),  # Annotation shape check (for empty)
                Flip_INbreast(orientation=parser.orientation),  # Flip images and annotations on orientation side
                Crop_INbreast(image_crop_shape=(parser.image_height_crop, parser.image_width_crop),
                              orientation=parser.orientation),  # Crop images and ROIs (same shape)
                # DATA
                Rescale_INbreast(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                Annotation_INbreast(),  # Annotation definition
                Add3ChannelsImage_INbreast(),  # Add 3 Channels to image [C, H, W]
                AnnotationPadding_INbreast(max_padding=parser.max_padding),  # Annotation padding (for batch dataloader)
                ToTensor_INbreast(),  # To Tensor
                # STANDARD NORMALIZATION
                StandardNormalization_INbreast(mean=std_statistics['train']['mean'], std=std_statistics['train']['std'])  # standard normalization
            ])

            # validation dataset transforms
            val_transforms = transforms.Compose([
                # PRE-PROCESSING
                SelectROI_INbreast(roi_type='Calcification'),  # select ROIs (annotations) type 'Calcification'
                AnnotationCheck_INbreast(),  # Annotation shape check (for empty)
                Flip_INbreast(orientation=parser.orientation),  # Flip all images and annotations on orientation side
                Crop_INbreast(image_crop_shape=(parser.image_height_crop, parser.image_width_crop),
                              orientation=parser.orientation),  # Crop all images and annotations to have same shape
                # DATA EVALUATION (only in inference)
                RadiusFilter_INbreast(radius=7),  # ignore calcifications with radius >= 7 pixel
                # DATA
                Rescale_INbreast(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                Annotation_INbreast(),  # Annotation definition
                Add3ChannelsImage_INbreast(),  # Add 3 Channels to image [C, H, W]
                AnnotationPadding_INbreast(max_padding=parser.max_padding),  # Annotation padding (for batch dataloader)
                ToTensor_INbreast(),  # To Tensor
                # STANDARD NORMALIZATION
                StandardNormalization_INbreast(mean=std_statistics['validation']['mean'], std=std_statistics['validation']['std'])  # standard normalization
            ])

            # test dataset transforms
            test_transforms = transforms.Compose([
                # PRE-PROCESSING
                SelectROI_INbreast(roi_type='Calcification'),  # select ROIs (annotations) type 'Calcification'
                AnnotationCheck_INbreast(),  # Annotation shape check (for empty)
                Flip_INbreast(orientation=parser.orientation),  # Flip all images and annotations on orientation side
                Crop_INbreast(image_crop_shape=(parser.image_height_crop, parser.image_width_crop),
                              orientation=parser.orientation),  # Crop all images and ROIs to have same shape
                # DATA EVALUATION (only in inference)
                RadiusFilter_INbreast(radius=7),  # ignore calcifications with radius >= 7 pixel
                # DATA
                Rescale_INbreast(rescale=parser.rescale),  # Rescale all images and annotations
                # DATA PREPARATION
                Annotation_INbreast(),  # Annotation definition
                Add3ChannelsImage_INbreast(),  # Add 3 Channels to image [C, H, W]
                AnnotationPadding_INbreast(max_padding=parser.max_padding),  # Annotation padding (for batch dataloader)
                ToTensor_INbreast(),  # To Tensor
                # STANDARD NORMALIZATION
                StandardNormalization_INbreast(mean=std_statistics['test']['mean'], std=std_statistics['test']['std'])  # standard normalization
            ])

        else:
            str_err = msg_error(file=__file__,
                                variable=normalization,
                                type_variable="normalization type",
                                choices="[none, min-max, std]")
            sys.exit(str_err)

    else:
        str_err = msg_error(file=__file__,
                            variable=parser.dataset,
                            type_variable="dataset",
                            choices="[E-ophtha-MA, INbreast]")

        sys.exit(str_err)

    return train_transforms, val_transforms, test_transforms
