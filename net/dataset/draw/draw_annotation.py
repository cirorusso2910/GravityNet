import cv2
import numpy as np

from net.dataset.draw.add_3_channels_image import add_3_channels_image
from net.dataset.utility.viewable_image import viewable_image


def draw_annotation(image: np.ndarray,
                    annotation: np.ndarray,
                    with_radius: bool,
                    draw_path: str):
    """
    Draw annotation on image and save

    :param image: image
    :param annotation: annotation
    :param with_radius: draw with radius option
    :param draw_path: path to save
    """

    image_3c = add_3_channels_image(image=image)
    image_3c = viewable_image(image=image_3c)

    num_annotation = annotation.shape[0]

    if num_annotation > 0:
        roi_x = annotation[:, 1]
        roi_y = annotation[:, 2]

        if annotation.shape[1] == 4:
            roi_r = annotation[:, 3]

        else:
            roi_r = np.ones(annotation.shape[0])

        for r in range(num_annotation):
            x = int(roi_x[r])
            y = int(roi_y[r])
            r = int(roi_r[r])

            if with_radius:
                cv2.circle(img=image_3c, center=(x, y), radius=r, color=(0, 255, 255), thickness=2)
            else:
                cv2.circle(img=image_3c, center=(x, y), radius=1, color=(0, 255, 255), thickness=2)

        cv2.imwrite(draw_path, image_3c)
