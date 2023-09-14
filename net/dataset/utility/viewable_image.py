import numpy as np


def viewable_image(image: np.ndarray) -> np.ndarray:
    """
    Transforms the input image in a 'viewable' image in the range [0, 255]

    :param image: image
    :return: 'viewable' image
    """

    image = image.copy()
    image = (image - image.min()) * 255 / (image.max() - image.min())
    image = image.astype(np.uint8)

    return image
