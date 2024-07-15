import numpy as np


class MyHorizontalFlip(object):
    """
    My Horizontal Flip: apply horizontal flip to sample data
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
        filename = sample['filename'] + "|HorizontalFlip"

        # flip image horizontal
        if self.num_channels == 1:
            image_horizontal_flip = np.fliplr(m=sample['image'])
        elif self.num_channels == 3:
            image_horizontal_flip = np.zeros(sample['image'].shape)
            image_horizontal_flip[0] = np.fliplr(m=sample['image'][0])
            image_horizontal_flip[1] = np.fliplr(m=sample['image'][1])
            image_horizontal_flip[2] = np.fliplr(m=sample['image'][2])
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

        # flip image mask horizontal
        image_mask_horizontal_flip = np.fliplr(m=sample['image_mask'])

        # flip annotation horizontal
        annotation_horizontal_flip = sample['annotation'].copy()
        annotation_horizontal_flip_length = len(annotation_horizontal_flip)
        if len(annotation_horizontal_flip) > 0:
            annotation_horizontal_flip[:, 0] = width - sample['annotation'][:, 0] - 1  # H - [x]
        else:
            annotation_horizontal_flip = np.zeros((0, annotation_horizontal_flip_length))

        sample = {'filename': filename,
                  'image': image_horizontal_flip,
                  'image_mask': image_mask_horizontal_flip,
                  'annotation': annotation_horizontal_flip
                  }

        return sample
