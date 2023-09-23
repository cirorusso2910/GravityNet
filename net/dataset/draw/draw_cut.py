import cv2
import numpy as np

from net.dataset.draw.add_3_channels_image import add_3_channels_image
from net.dataset.utility.viewable_image import viewable_image


def draw_cut(image: np.ndarray,
             x_min_cut: int,
             y_min_cut: int,
             x_max_cut: int,
             y_max_cut: int,
             image_path: str):
    """
    Draw cut line on image and save

    :param image: image
    :param x_min_cut: coords x_min cut
    :param y_min_cut: coords y_min cut
    :param x_max_cut: coords x_max cut
    :param y_max_cut: coords y_max cut
    :param image_path: path to save
    """

    image_3c = add_3_channels_image(image=image)
    image_3c = viewable_image(image=image_3c)

    cv2.line(image_3c, (x_min_cut, y_min_cut), (x_min_cut + x_max_cut, y_min_cut), color=(0, 255, 0), thickness=2)
    cv2.line(image_3c, (x_min_cut, y_max_cut), (x_max_cut, y_max_cut), color=(0, 255, 0), thickness=2)
    cv2.line(image_3c, (x_max_cut, y_min_cut), (x_max_cut, y_max_cut), color=(0, 255, 0), thickness=2)

    cv2.imwrite(image_path, image_3c)
