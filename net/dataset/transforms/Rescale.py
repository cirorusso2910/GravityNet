import cv2
import numpy as np


class Rescale(object):
    """
    Rescale: rescale data according to rescale factor [0, 1]
    - rescale 1.0: no rescale
    """

    def __init__(self,
                 rescale: float,
                 num_channels: int):
        """
        __init__ method: run one when instantiating the object

        :param rescale: rescale factor
        :param num_channels: image num channels
        """

        self.rescale = rescale

        self.num_channels = num_channels

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # get image shape
        if self.num_channels == 1:
            # image shape (H x W)
            height, width = sample['image'].shape  # H x W
        elif self.num_channels == 3:
            # image shape (C x H x W)
            channels, height, width = sample['image'].shape  # C x H x W
        else:
            raise ValueError("Invalid number of channels: supported values are 1 (grayscale) and 3 (RGB).")

        # rescaled shape
        image_rescale_shape = (int(width * self.rescale), int(height * self.rescale))

        # rescale image
        if self.num_channels == 1:
            # rescale grayscale channel
            image_rescale = cv2.resize(sample['image'], image_rescale_shape, interpolation=cv2.INTER_AREA)

        elif self.num_channels == 3:
            # rescale each channel separately
            image_rescale_channels = [cv2.resize(sample['image'][i], image_rescale_shape, interpolation=cv2.INTER_AREA)
                                      for i in range(self.num_channels)]

            # merge channels
            image_rescale = cv2.merge(image_rescale_channels)

            # transpose to (C x W x W)
            image_rescale = np.transpose(image_rescale, (2, 0, 1))
        else:
            raise ValueError("Invalid number of channels: supported values are 1 (grayscale) and 3 (RGB).")

        # rescale image mask
        image_mask_rescale = cv2.resize(sample['image_mask'], image_rescale_shape, interpolation=cv2.INTER_AREA)

        # rescale annotation
        annotation_rescale = sample['annotation'].copy()
        num_annotation = len(annotation_rescale)
        for i in range(num_annotation):
            annotation_rescale[i][0] = int(sample['annotation'][i, 0] * self.rescale)
            annotation_rescale[i][1] = int(sample['annotation'][i, 1] * self.rescale)

        sample = {'filename': sample['filename'],
                  'image': image_rescale,
                  'image_mask': image_mask_rescale,
                  'annotation': annotation_rescale,
                  }

        return sample
