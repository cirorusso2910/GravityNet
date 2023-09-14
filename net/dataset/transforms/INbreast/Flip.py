import sys

import cv2
import numpy as np
import os


class Flip(object):
    """
    Flip data according to orientation side
    """

    def __init__(self,
                 orientation: str):
        """
        __init__ method: run one when instantiating the object

        :param orientation: orientation side [L, R]]
        :param debug: debug option
        """

        self.orientation = orientation

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # filename
        filename = sample['filename']

        # image orientation
        image_orientation = filename.split('_')[3]

        # if image orientation is the same of orientation: no flip
        if image_orientation == self.orientation:
            image_flip = sample['image']
            image_mask_flip = sample['image_mask']
            annotation_flip = sample['annotation']
            annotation_w48m14_flip = sample['annotation_w48m14']

        # else: flip
        else:

            # flip image
            image_flip = np.fliplr(sample['image'])

            # flip image-mask
            image_mask_flip = np.fliplr(sample['image_mask'])

            # image shape
            image_shape = sample['image'].shape

            # flip annotation
            annotation_flip = sample['annotation'].copy()
            if len(annotation_flip) > 0:
                annotation_flip[:, 1] = image_shape[1] - sample['annotation'][:, 1] - 1

            # flip annotation-w48m14
            annotation_w48m14_flip = sample['annotation_w48m14'].copy()
            if len(annotation_w48m14_flip) > 0:
                annotation_w48m14_flip[:, 1] = image_shape[1] - sample['annotation_w48m14'][:, 1] - 1

        sample = {'filename': sample['filename'],
                  'image': image_flip,
                  'image_mask': image_mask_flip,
                  'annotation': annotation_flip,
                  'annotation_w48m14': annotation_w48m14_flip
                  }

        return sample
