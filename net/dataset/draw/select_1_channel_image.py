import cv2
import numpy as np


def select_1_channel_image(image: np.ndarray) -> np.ndarray:
    """
    Add 3 channels to image: copy image 3 times

    :param image: image
    :return: image with 3 channels
    """

    image = np.transpose(image, (1, 2, 0))  # CxHxW -> HxWxC
    image = cv2.merge((image[:, :, 2], image[:, :, 1], image[:, :, 0]))  # merge channels

    return image
