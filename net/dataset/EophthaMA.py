import os

from pandas import read_csv
from skimage import io
from typing import Any, List
from torch.utils.data import Dataset

from net.initialization.header.annotations import annotation_header


class EophthaMA(Dataset):
    """
    E-ophtha-MA Dataset
    """

    def __init__(self,
                 images_dir: str,
                 images_masks_dir: str,
                 annotations_dir: str,
                 filename_list: List[str],
                 transforms: Any):
        """
        __init__ method: run one when instantiating the object

        :param images_dir: images directory
        :param images_masks_dir: images masks directory
        :param annotations_dir: annotations directory
        :param filename_list: filename list
        :param transforms: transforms dataset to apply
        """

        self.images_dir = images_dir
        self.images_masks_dir = images_masks_dir
        self.annotations_dir = annotations_dir
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
        image_filename = self.filename_list[idx] + ".cropped.jpg"
        image_path = os.path.join(self.images_dir, image_filename)
        image = io.imread(image_path)  # numpy.ndarray

        # ---------- #
        # IMAGE MASK #
        # ---------- #
        image_mask_filename = self.filename_list[idx] + ".mask.cropped.png"
        image_mask_path = os.path.join(self.images_masks_dir, image_mask_filename)
        image_mask = io.imread(image_mask_path)  # numpy.ndarray

        # ---------- #
        # ANNOTATION #
        # ---------- #
        annotation_filename = self.filename_list[idx] + ".cropped.csv"
        annotation_path = os.path.join(self.annotations_dir, annotation_filename)
        annotation = read_csv(filepath_or_buffer=annotation_path, usecols=annotation_header(dataset='E-ophtha-MA', annotation_type='default')).values

        sample = {'filename': self.filename_list[idx],
                  'image': image,
                  'image_mask': image_mask,
                  'annotation': annotation,
                  }

        if self.transforms:
            sample = self.transforms(sample)

        return sample
