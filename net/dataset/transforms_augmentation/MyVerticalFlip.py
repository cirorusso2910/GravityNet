import numpy as np


class MyVerticalFlip(object):
    """
    My Vertical Flip: apply vertical flip to sample data
    """

    def __init__(self,
                 num_channels: int):
        """
        __init__ method: run one when instantiating the object

        :param num_channels: image num channels
        """

        self.num_channels = num_channels

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # filename
        filename = sample['filename'] + "|VerticalFlip"

        # flip image vertical
        if self.num_channels == 1:
            image_vertical_flip = np.flipud(m=sample['image'])
        elif self.num_channels == 3:
            image_vertical_flip = np.zeros(sample['image'].shape)
            image_vertical_flip[0] = np.flipud(m=sample['image'][0])
            image_vertical_flip[1] = np.flipud(m=sample['image'][1])
            image_vertical_flip[2] = np.flipud(m=sample['image'][2])
        else:
            raise ValueError("Invalid number of channels: supported values are 1 (grayscale) and 3 (RGB).")

        # get image shape
        if self.num_channels == 1:
            # image shape (H x W)
            height, width = sample['image'].shape  # H x W
        elif self.num_channels == 3:
            # image shape (C x H x W)
            channels, height, width = sample['image'].shape  # C x H x W
        else:
            raise ValueError("Invalid number of channels: supported values are 1 (grayscale) and 3 (RGB).")

        # flip image mask vertical
        image_mask_vertical_flip = np.flipud(m=sample['image_mask'])

        # flip annotation vertical
        annotation_vertical_flip = sample['annotation'].copy()
        annotation_vertical_flip_length = len(annotation_vertical_flip)
        if len(annotation_vertical_flip) > 0:
            annotation_vertical_flip[:, 1] = height - sample['annotation'][:, 1] - 1  # W - [y]
        else:
            annotation_vertical_flip = np.zeros((0, annotation_vertical_flip_length))

        sample = {'filename': filename,
                  'image': image_vertical_flip,
                  'image_mask': image_mask_vertical_flip,
                  'annotation': annotation_vertical_flip
                  }

        return sample
