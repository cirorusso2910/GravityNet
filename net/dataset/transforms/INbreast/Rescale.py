import sys

import cv2
import os


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
        :param debug: debug option
        """

        self.rescale = rescale

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # image shape (H x W)
        image_shape = sample['image'].shape
        image_w48m14_shape = sample['image_w48m14'].shape

        # rescale shape (W x H)
        image_rescale_shape = (int(image_shape[1] * self.rescale), int(image_shape[0] * self.rescale))
        image_w48m14_rescale_shape = (int(image_w48m14_shape[1] * self.rescale), int(image_w48m14_shape[0] * self.rescale))

        # rescale image
        image_rescale = cv2.resize(sample['image'], image_rescale_shape, interpolation=cv2.INTER_AREA)
        image_w48m14_rescale = cv2.resize(sample['image_w48m14'], image_w48m14_rescale_shape, interpolation=cv2.INTER_AREA)

        # rescale image mask
        image_mask_rescale = cv2.resize(sample['image_mask'], image_rescale_shape, interpolation=cv2.INTER_AREA)
        image_mask_w48m14_rescale = cv2.resize(sample['image_mask_w48m14'], image_w48m14_rescale_shape, interpolation=cv2.INTER_AREA)

        # rescale annotation
        annotation_rescale = sample['annotation'].copy()
        annotation_w48m14_rescale = sample['annotation_w48m14'].copy()

        num_annotation = len(annotation_rescale)
        num_annotation_w48m14 = len(annotation_w48m14_rescale)

        for i in range(num_annotation):
            annotation_rescale[i][1] = int(sample['annotation'][i, 1] * self.rescale)
            annotation_rescale[i][2] = int(sample['annotation'][i, 2] * self.rescale)
            annotation_rescale[i][3] = int(sample['annotation'][i, 3] * self.rescale) if annotation_rescale[i][3] > 1 else 1

        for i in range(num_annotation_w48m14):
            annotation_w48m14_rescale[i][1] = int(sample['annotation_w48m14'][i, 1] * self.rescale)
            annotation_w48m14_rescale[i][2] = int(sample['annotation_w48m14'][i, 2] * self.rescale)

        sample = {'filename': sample['filename'],
                  'image': image_rescale,
                  'image_w48m14': image_w48m14_rescale,
                  'image_mask': image_mask_rescale,
                  'image_mask_w48m14': image_mask_w48m14_rescale,
                  'annotation': annotation_rescale,
                  'annotation_w48m14': annotation_w48m14_rescale
                  }

        return sample
