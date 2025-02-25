import numpy as np


def add_3_channels_image(image: np.ndarray) -> np.ndarray:
    """
    Add 3 channels to image: copy image 3 times

    :param image: image
    :return: image with 3 channels
    """

    # init
    image_3c = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # copy image 3 times
    image_3c[:, :, 0] = image
    image_3c[:, :, 1] = image
    image_3c[:, :, 2] = image

    return image_3c
