import sys

import cv2
import numpy as np

from net.dataset.draw.add_3_channels_image import add_3_channels_image
from net.utility.msg.msg_error import msg_error


def select_image_channel(image: np.ndarray,
                         channel: str) -> np.ndarray:
    """
    Select image channel

    :param image: image
    :param channel: channel
    :return: image with channel selected
    """

    # split image BGR channel: B (Blue) | G (Green) | R (Red)
    # NOTE: in case of channel G, three channels are equal (green)
    (image_blue_channel, image_green_channel, image_red_channel) = cv2.split(image)

    # image blue channel
    if channel == 'B':
        image_channel = image_blue_channel
        # copy image channel 3 times (because image MUST have 3 channels)
        image_3_channels = add_3_channels_image(image=image_channel)

    # image green channel
    elif channel == 'G':
        image_channel = image_green_channel
        # copy image channel 3 times (because image MUST have 3 channels)
        image_3_channels = add_3_channels_image(image=image_channel)

    # image red channel
    elif channel == 'R':
        image_channel = image_red_channel
        # copy image channel 3 times (because image MUST have 3 channels)
        image_3_channels = add_3_channels_image(image=image_channel)

    elif channel == 'RGB' or 'BGR':
        image_3_channels = cv2.merge((image[:, :, 2], image[:, :, 1], image[:, :, 0]))

    else:
        str_err = msg_error(file=__file__,
                            variable=channel,
                            type_variable='image channel',
                            choices='[B, G, R, RGB/BGR]')
        sys.exit(str_err)

    return image_3_channels
