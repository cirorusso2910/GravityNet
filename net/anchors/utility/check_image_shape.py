import numpy as np


def check_image_shape(image_shape: np.array):
    """
    Check if image shape dimension is multiple of 32.
    Otherwise, the gravity points are not placed within the image.

    :param image_shape: image shape
    """

    image_height, image_width = image_shape

    if image_height % 32 != 0 and image_width % 32 != 0:
        print("Image shape is NOT multiple of 32"
              "\nNOTE: the gravity points are not placed within the image")