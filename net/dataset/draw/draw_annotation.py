import cv2
import numpy as np

from net.colors.colors import *
from net.dataset.draw.add_3_channels_image import add_3_channels_image
from net.dataset.draw.transpose_3_channel_image import transpose_3_channel_image
from net.utility.msg.msg_error import msg_error


def draw_annotation(image: np.ndarray,
                    annotation: np.ndarray,
                    type: str,
                    image_path: str):
    """
    Draw annotation on image and save

    :param image: image
    :param annotation: annotation
    :param type: type of draw
    :param image_path: path to save
    """

    if image.shape[0] == 3:  # image with 3 channels
        image = transpose_3_channel_image(image=image)
    else:
        image = add_3_channels_image(image=image)

    num_annotation = annotation.shape[0]

    if num_annotation > 0:
        roi_x = annotation[:, 0]
        roi_y = annotation[:, 1]

        for r in range(num_annotation):
            x = int(roi_x[r])
            y = int(roi_y[r])

            # draw cross
            if type == 'cross':
                cv2.line(image, (x - 10, y - 10), (x + 10, y + 10), color=YELLOW1, thickness=1)
                cv2.line(image, (x - 10, y + 10), (x + 10, y - 10), color=YELLOW1, thickness=1)
                # cv2.drawMarker(img=image, position=(x, y), color=YELLOW1, markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.imwrite(image_path, image)

            # draw point
            elif type == 'point':
                cv2.circle(img=image, center=(x, y), radius=0, color=YELLOW1, thickness=-1)
                cv2.imwrite(image_path, image)

            # draw square
            elif type == 'square':
                start_point = (x - 10, y - 10)  # start point (top left corner of rectangle)
                end_point = (x + 10, y + 10)  # end point (bottom right corner of rectangle)# draw prediction (TP)
                cv2.rectangle(img=image, pt1=start_point, pt2=end_point, color=YELLOW1, thickness=1)
                cv2.imwrite(image_path, image)

            else:
                msg_error(file=__file__,
                          variable=type,
                          type_variable='Type Draw',
                          choices='[cross, point, square]')
