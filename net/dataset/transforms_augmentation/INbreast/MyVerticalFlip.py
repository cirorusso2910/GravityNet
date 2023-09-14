import numpy as np


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
        image_vertical_flip = np.flipud(m=sample['image'])
        # image w48m14 vertical flip
        image_w48m14_vertical_flip = np.flipud(m=sample['image_w48m14'])

        # image mask vertical flip
        image_mask_vertical_flip = np.flipud(m=sample['image_mask'])
        # image mask w48m14 vertical flip
        image_mask_w48m14_vertical_flip = np.flipud(m=sample['image_mask_w48m14'])

        # image shape [H x W]
        image_shape = sample['image'].shape
        image_w48m14_shape = sample['image'].shape

        # annotation vertical flip
        annotation_vertical_flip = sample['annotation'].copy()
        if len(annotation_vertical_flip) > 0:
            annotation_vertical_flip[:, 2] = image_shape[0] - sample['annotation'][:, 2] - 1
        else:
            annotation_vertical_flip = np.zeros((0, 3))

        # annotation w48m14 vertical flip
        annotation_w48m14_vertical_flip = sample['annotation_w48m14'].copy()
        if len(annotation_w48m14_vertical_flip) > 0:
            annotation_w48m14_vertical_flip[:, 2] = image_w48m14_shape[0] - sample['annotation_w48m14'][:, 2] - 1
        else:
            annotation_w48m14_vertical_flip = np.zeros((0, 2))

        sample = {'filename': filename,
                  'image': image_vertical_flip,
                  'image_w48m14': image_w48m14_vertical_flip,
                  'image_mask': image_mask_vertical_flip,
                  'image_mask_w48m14': image_mask_w48m14_vertical_flip,
                  'annotation': annotation_vertical_flip,
                  'annotation_w48m14': annotation_w48m14_vertical_flip
                  }

        return sample
