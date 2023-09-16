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
        image_horizontal_flip = np.zeros(sample['image'].shape)
        image_horizontal_flip[0] = np.fliplr(m=sample['image'][0])
        image_horizontal_flip[1] = np.fliplr(m=sample['image'][1])
        image_horizontal_flip[2] = np.fliplr(m=sample['image'][2])

        # image mask horizontal flip
        image_mask_horizontal_flip = np.fliplr(m=sample['image_mask'])

        # image shape [C x H x W]
        image_shape = sample['image'].shape

        # annotation horizontal flip
        annotation_horizontal_flip = np.zeros(sample['annotation'].shape)
        if len(annotation_horizontal_flip) > 0:
            annotation_horizontal_flip[:, 0] = image_shape[2] - sample['annotation'][:, 0] - 1  # H - [x]
        else:
            annotation_horizontal_flip = np.zeros((0, 4))

        sample = {'filename': filename,
                  'image': image_horizontal_flip,
                  'image_mask': image_mask_horizontal_flip,
                  'annotation': annotation_horizontal_flip
                  }

        return sample
