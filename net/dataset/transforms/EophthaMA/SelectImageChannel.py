import cv2
import numpy as np
import os
import sys

from net.utility.msg.msg_error import msg_error


class SelectImageChannel(object):
    """
    Select image channel [RGB or G]
    """

    def __init__(self,
                 channel: str):
        """
        __init__ method: run one when instantiating the object

        :param channel: channel
        :param debug:  debug option
        """

        self.channel = channel

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        filename = sample['filename']
        image = sample['image']
        annotation = sample['annotation']

        if self.channel == 'RGB':
            image_channels = sample['image']

        elif self.channel == 'G':
            # green channel [image BGR -> G]
            new_image_G = sample['image'][1, :, :]

            # copy green channel
            image_channels = np.zeros(image.shape)
            image_channels[0] = new_image_G
            image_channels[1] = new_image_G
            image_channels[2] = new_image_G

        else:
            str_err = msg_error(file=__file__,
                                variable=self.channel,
                                type_variable='channel',
                                choices='[BGR, G]')
            sys.exit(str_err)

        sample = {'filename': sample['filename'],
                  'image': image_channels,
                  'image_mask': sample['image_mask'],
                  'annotation': sample['annotation']
                  }

        return sample
