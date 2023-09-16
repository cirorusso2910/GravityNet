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
        image_horizontal_flip = np.zeros(sample['image'].shape)
        image_horizontal_flip[0] = np.fliplr(m=sample['image'][0])
        image_horizontal_flip[1] = np.fliplr(m=sample['image'][1])
        image_horizontal_flip[2] = np.fliplr(m=sample['image'][2])
        image_horizontal_and_vertical_flip = np.zeros(sample['image'].shape)
        image_horizontal_and_vertical_flip[0] = np.flipud(m=image_horizontal_flip[0])
        image_horizontal_and_vertical_flip[1] = np.flipud(m=image_horizontal_flip[1])
        image_horizontal_and_vertical_flip[2] = np.flipud(m=image_horizontal_flip[2])

        # image mask horizontal and vertical flip
        image_mask_horizontal_flip = np.fliplr(m=sample['image_mask'])
        image_mask_horizontal_and_vertical_flip = np.flipud(m=image_mask_horizontal_flip)

        # image shape [C x H x W]
        image_shape = sample['image'].shape

        # annotation vertical flip
        annotation_horizontal_and_vertical_flip = np.zeros(sample['annotation'].shape)
        if len(annotation_horizontal_and_vertical_flip) > 0:
            annotation_horizontal_and_vertical_flip[:, 0] = image_shape[2] - sample['annotation'][:, 0] - 1
            annotation_horizontal_and_vertical_flip[:, 1] = image_shape[1] - sample['annotation'][:, 1] - 1
            annotation_horizontal_and_vertical_flip[:, 2] = sample['annotation'][:, 2]
            annotation_horizontal_and_vertical_flip[:, 3] = sample['annotation'][:, 3]
        else:
            annotation_horizontal_and_vertical_flip = np.zeros((0, 4))

        sample = {'filename': filename,
                  'image': image_horizontal_and_vertical_flip,
                  'image_mask': image_mask_horizontal_and_vertical_flip,
                  'annotation': annotation_horizontal_and_vertical_flip
                  }

        return sample
