import numpy as np


class Add3ChannelsImage(object):
    """
    Add 3 Channels to Image: add 3 channels to image (copying 3 times)
    """

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # read image
        image = sample['image']  # H x W
        image_48m14 = sample['image_w48m14']  # H x W

        # add 3 channels to image
        image_3c = np.zeros((3, image.shape[0], image.shape[1]))  # C x H x W
        image_48m14_3c = np.zeros((3, image_48m14.shape[0], image_48m14.shape[1]))  # C x H x W

        # copy the content of the image_original in each channel
        image_3c[0, :, :] = image
        image_3c[1, :, :] = image
        image_3c[2, :, :] = image

        # copy the content of the image_original in each channel
        image_48m14_3c[0, :, :] = image_48m14
        image_48m14_3c[1, :, :] = image_48m14
        image_48m14_3c[2, :, :] = image_48m14

        sample = {'filename': sample['filename'],
                  'image': image_3c,
                  'image_w48m14': image_48m14_3c,
                  'image_mask': sample['image_mask'],
                  'image_mask_w48m14': sample['image_mask_w48m14'],
                  'annotation': sample['annotation'],
                  'annotation_w48m14': sample['annotation_w48m14']
                  }

        return sample
