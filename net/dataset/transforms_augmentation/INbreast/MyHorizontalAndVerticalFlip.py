import numpy as np


class MyHorizontalAndVerticalFlip(object):
    """
    My Horizontal and Vertical Flip: apply horizontal and vertical flip to sample data
    """

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # filename
        filename = sample['filename'] + "|HorizontalAndVerticalFlip"

        # image horizontal and vertical flip
        image_horizontal_flip = np.fliplr(m=sample['image'])
        image_horizontal_and_vertical_flip = np.flipud(m=image_horizontal_flip)
        # image w48m14 horizontal and vertical flip
        image_w48m14_horizontal_flip = np.fliplr(m=sample['image_w48m14'])
        image_w48m14_horizontal_and_vertical_flip = np.flipud(m=image_w48m14_horizontal_flip)

        # image mask horizontal and vertical flip
        image_mask_horizontal_flip = np.fliplr(m=sample['image_mask'])
        image_mask_horizontal_and_vertical_flip = np.flipud(m=image_mask_horizontal_flip)
        # image mask w48m14 horizontal and vertical flip
        image_mask_w48m14_horizontal_flip = np.fliplr(m=sample['image_mask_w48m14'])
        image_mask_w48m14_horizontal_and_vertical_flip = np.flipud(m=image_mask_w48m14_horizontal_flip)

        # image shape
        image_shape = sample['image'].shape
        image_w48m14_shape = sample['image'].shape

        # annotation horizontal and vertical flip
        annotation_horizontal_and_vertical_flip = sample['annotation'].copy()
        if len(annotation_horizontal_and_vertical_flip) > 0:
            annotation_horizontal_and_vertical_flip[:, 2] = image_shape[0] - sample['annotation'][:, 2] - 1
            annotation_horizontal_and_vertical_flip[:, 1] = image_shape[1] - sample['annotation'][:, 1] - 1
        else:
            annotation_horizontal_and_vertical_flip = np.zeros((0, 3))

        # annotation w48m14 horizontal and vertical flip
        annotation_w48m14_horizontal_and_vertical_flip = sample['annotation_w48m14'].copy()
        if len(annotation_w48m14_horizontal_and_vertical_flip) > 0:
            annotation_w48m14_horizontal_and_vertical_flip[:, 2] = image_w48m14_shape[0] - sample['annotation_w48m14'][:, 2] - 1
            annotation_w48m14_horizontal_and_vertical_flip[:, 1] = image_w48m14_shape[1] - sample['annotation_w48m14'][:, 1] - 1
        else:
            annotation_w48m14_horizontal_and_vertical_flip = np.zeros((0, 2))

        sample = {'filename': filename,
                  'image': image_horizontal_and_vertical_flip,
                  'image_w48m14': image_w48m14_horizontal_and_vertical_flip,
                  'image_mask': image_mask_horizontal_and_vertical_flip,
                  'image_mask_w48m14': image_mask_w48m14_horizontal_and_vertical_flip,
                  'annotation': annotation_horizontal_and_vertical_flip,
                  'annotation_w48m14': annotation_w48m14_horizontal_and_vertical_flip
                  }

        return sample
