import numpy as np


class MyHorizontalFlip(object):
    """
    My Horizontal Flip: apply horizontal flip to sample data
    """

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # filename
        filename = sample['filename'] + "|HorizontalFlip"

        # image horizontal flip
        image_horizontal_flip = np.fliplr(m=sample['image'])
        # image w48m14 horizontal flip
        image_w48m14_horizontal_flip = np.fliplr(m=sample['image_w48m14'])

        # image mask horizontal flip
        image_mask_horizontal_flip = np.fliplr(m=sample['image_mask'])
        # image mask w48m14 horizontal flip
        image_mask_w48m14_horizontal_flip = np.fliplr(m=sample['image_mask_w48m14'])

        # image shape
        image_shape = sample['image'].shape
        image_w48m14_shape = sample['image'].shape

        # annotation horizontal flip
        annotation_horizontal_flip = sample['annotation'].copy()
        if len(annotation_horizontal_flip) > 0:
            annotation_horizontal_flip[:, 1] = image_shape[1] - sample['annotation'][:, 1] - 1  # H - [x]
        else:
            annotation_horizontal_flip = np.zeros((0, 3))

        # annotation w48m14 horizontal flip
        annotation_w48m14_horizontal_flip = sample['annotation_w48m14'].copy()
        if len(annotation_w48m14_horizontal_flip) > 0:
            annotation_w48m14_horizontal_flip[:, 1] = image_w48m14_shape[1] - sample['annotation_w48m14'][:, 1] - 1  # H - [x]
        else:
            annotation_w48m14_horizontal_flip = np.zeros((0, 2))

        sample = {'filename': filename,
                  'image': image_horizontal_flip,
                  'image_w48m14': image_w48m14_horizontal_flip,
                  'image_mask': image_mask_horizontal_flip,
                  'image_mask_w48m14': image_mask_w48m14_horizontal_flip,
                  'annotation': annotation_horizontal_flip,
                  'annotation_w48m14': annotation_w48m14_horizontal_flip
                  }

        return sample
