import cv2
import numpy as np
import os
import sys


class MyVerticalFlip(object):
    """
    My Vertical Flip: apply vertical flip to sample data
    """

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # filename
        filename = sample['filename'] + "|VerticalFlip"

        # image vertical flip
        image_vertical_flip = np.zeros(sample['image'].shape)
        image_vertical_flip[0] = np.flipud(m=sample['image'][0])
        image_vertical_flip[1] = np.flipud(m=sample['image'][1])
        image_vertical_flip[2] = np.flipud(m=sample['image'][2])

        # image mask vertical flip
        image_mask_vertical_flip = np.flipud(m=sample['image_mask'])

        # image shape [C x H x W]
        image_shape = sample['image'].shape

        # annotation vertical flip
        annotation_vertical_flip = np.zeros(sample['annotation'].shape)
        if len(annotation_vertical_flip) > 0:
            annotation_vertical_flip[:, 0] = sample['annotation'][:, 0]
            annotation_vertical_flip[:, 1] = image_shape[1] - sample['annotation'][:, 1] - 1
            annotation_vertical_flip[:, 2] = sample['annotation'][:, 2]
            annotation_vertical_flip[:, 3] = sample['annotation'][:, 3]
        else:
            annotation_vertical_flip = np.zeros((0, 4))

        sample = {'filename': filename,
                  'image': image_vertical_flip,
                  'image_mask': image_mask_vertical_flip,
                  'annotation': annotation_vertical_flip
                  }

        return sample
