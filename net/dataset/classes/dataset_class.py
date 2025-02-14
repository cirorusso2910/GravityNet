import os

import numpy as np
from pandas import read_csv
from skimage import io
from typing import Any, List
from torch.utils.data import Dataset

from net.initialization.header.annotations import annotation_header


class dataset_class(Dataset):
    """
    Class Dataset
    """

    def __init__(self,
                 images_dir: str,
                 images_extension: str,
                 images_masks_dir: str,
                 images_masks_extension: str,
                 annotations_dir: str,
                 annotations_extension: str,
                 filename_list: List[str],
                 transforms: Any):
        """
        __init__ method: run one when instantiating the object

        :param images_dir: images directory
        :param images_extension: images file extension
        :param images_masks_dir: images masks directory
        :param images_masks_extension: images masks file extension
        :param annotations_dir: annotations directory
        :param annotations_extension: annotations file extension
        :param filename_list: filename list
        :param transforms: transforms dataset to apply
        """

        self.images_dir = images_dir
        self.images_extension = images_extension
        self.images_masks_dir = images_masks_dir
        self.images_masks_extension = images_masks_extension
        self.annotations_dir = annotations_dir
        self.annotations_extension = annotations_extension
        self.filename_list = filename_list
        self.transforms = transforms

    def __len__(self) -> int:
        """
        __len__ method: returns the number of samples in dataset

        :return: number of samples in dataset
        """

        return len(self.filename_list)

    def __getitem__(self,
                    idx: int) -> dict:
        """
        __getitem__ method: loads and return a sample from the dataset at given index

        :param idx: sample index
        :return: sample dictionary
        """

        # ----- #
        # IMAGE #
        # ----- #
        image_filename = self.filename_list[idx] + ".{}".format(self.images_extension)
        image_path = os.path.join(self.images_dir, image_filename)
        image = io.imread(image_path)  # numpy.ndarray

        # ---------- #
        # IMAGE MASK #
        # ---------- #
        if self.images_masks_extension != 'none':
            image_mask_filename = self.filename_list[idx] + ".mask.{}".format(self.images_masks_extension)
            image_mask_path = os.path.join(self.images_masks_dir, image_mask_filename)
            if os.path.isfile(image_mask_path):
                image_mask = io.imread(image_mask_path)  # numpy.ndarray
            else:
                # define full image mask
                image_mask = np.full(shape=(image.shape[0], image.shape[1]), fill_value=255, dtype=np.uint8)
                # print("WARNING: full image mask")
        else:
            # do not consider image mask
            image_mask = np.full(shape=(image.shape[0], image.shape[1]), fill_value=255, dtype=np.uint8)

        # ---------- #
        # ANNOTATION #
        # ---------- #
        annotation_filename = self.filename_list[idx] + ".{}".format(self.annotations_extension)
        annotation_path = os.path.join(self.annotations_dir, annotation_filename)
        header = annotation_header(annotation_type='default')
        annotation = read_csv(filepath_or_buffer=annotation_path, usecols=header).values

        sample = {'filename': self.filename_list[idx],
                  'image': image,
                  'image_mask': image_mask,
                  'annotation': annotation,
                  }

        if self.transforms:
            sample = self.transforms(sample)

        return sample
