import os

from pandas import read_csv
from skimage import io
from torch.utils.data import Dataset
from typing import List, Any

from net.initialization.header.annotations import annotation_header


class INbreast(Dataset):
    """
    INbreast Dataset
    """

    def __init__(self,
                 images_dir: str,
                 images_masks_dir: str,
                 annotations_dir: str,
                 annotations_w48m14_dir: str,
                 filename_list: List[str],
                 transforms: Any):
        """
        __init__ method: run one when instantiating the object

        :param images_dir: images directory
        :param images_masks_dir: images maks directory
        :param annotations_dir: annotations directory
        :param annotations_w48m14_dir: annotations w48m14 directory
        :param filename_list: filename list
        :param transforms: transforms dataset to apply
        """

        self.images_dir = images_dir
        self.images_masks_dir = images_masks_dir
        self.annotations_dir = annotations_dir
        self.annotations_w48m14_dir = annotations_w48m14_dir
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
        image_filename = self.filename_list[idx] + ".tif"
        image_path = os.path.join(self.images_dir, image_filename)
        image = io.imread(image_path)  # numpy.ndarray

        # ---------- #
        # IMAGE MASK #
        # ---------- #
        image_mask_filename = self.filename_list[idx] + ".mask.png"
        image_mask_path = os.path.join(self.images_masks_dir, image_mask_filename)
        image_mask = io.imread(image_mask_path)  # numpy.ndarray

        # ---------- #
        # ANNOTATION #
        # ---------- #
        annotation_filename = self.filename_list[idx] + ".csv"
        annotation_path = os.path.join(self.annotations_dir, annotation_filename)
        annotation = read_csv(filepath_or_buffer=annotation_path, usecols=annotation_header(dataset='INbreast', annotation_type='default')).values  # numpy.ndarray

        # ----------------- #
        # ANNOTATION w48m14 #
        # ----------------- #
        # annotation most suitable for training
        annotation_w48m14_filename = self.filename_list[idx] + ".csv"
        annotation_w48m14_path = os.path.join(self.annotations_w48m14_dir, annotation_w48m14_filename)
        annotation_w48m14 = read_csv(filepath_or_buffer=annotation_w48m14_path, usecols=annotation_header(dataset='INbreast', annotation_type='w48m14')).values  # numpy.ndarray

        sample = {'filename': self.filename_list[idx],
                  'image': image,
                  'image_mask': image_mask,
                  'annotation': annotation,
                  'annotation_w48m14': annotation_w48m14
                  }

        if self.transforms:
            sample = self.transforms(sample)

        return sample
