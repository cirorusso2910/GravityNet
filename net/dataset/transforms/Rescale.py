import cv2

import numpy as np


class Rescale(object):
    """
    Rescale data according to rescale factor (from 0 to 1)
        - rescale 1.0: no rescale
    """

    def __init__(self,
                 rescale: float):
        """
        __init__ method: run one when instantiating the object

        :param rescale: rescale factor
        """

        self.rescale = rescale

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # image shape (C x H x W)
        channels, height, width = sample['image'].shape

        # rescale shape (W x H)
        image_rescale_shape = (int(width * self.rescale), int(height * self.rescale))

        # rescale image
        image_rescale_B = cv2.resize(sample['image'][0], image_rescale_shape, interpolation=cv2.INTER_AREA)  # rescale image blue channel  [2]
        image_rescale_G = cv2.resize(sample['image'][1], image_rescale_shape, interpolation=cv2.INTER_AREA)  # rescale image green channel [1]
        image_rescale_R = cv2.resize(sample['image'][2], image_rescale_shape, interpolation=cv2.INTER_AREA)  # rescale image red channel [0]
        image_rescale = cv2.merge((image_rescale_B, image_rescale_G, image_rescale_R))  # merge channels [BGR]
        image_rescale = np.transpose(image_rescale, (2, 0, 1))  # transpose HxWxC -> CxHxW

        # rescale image mask
        image_mask_rescale = cv2.resize(sample['image_mask'], image_rescale_shape, interpolation=cv2.INTER_AREA)

        # rescale annotation
        annotation_rescale = sample['annotation'].copy()

        num_annotation = len(annotation_rescale)

        for i in range(num_annotation):
            annotation_rescale[i][0] = int(sample['annotation'][i, 0] * self.rescale)
            annotation_rescale[i][1] = int(sample['annotation'][i, 1] * self.rescale)
            annotation_rescale[i][2] = int(sample['annotation'][i, 2] * self.rescale) if annotation_rescale[i][2] > 1 else 1
            annotation_rescale[i][3] = int(sample['annotation'][i, 3] * self.rescale) if annotation_rescale[i][3] > 1 else 1

        sample = {'filename': sample['filename'],
                  'image': image_rescale,
                  'image_mask': image_mask_rescale,
                  'annotation': annotation_rescale,
                  }

        return sample
