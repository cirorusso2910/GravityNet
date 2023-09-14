import cv2
import numpy as np
import os
import sys

from typing import Tuple


class Crop(object):
    """
    Crop data according to image crop shape

    INbreast has two mammogram sizes: 4084 x 3328 and 3328 x 2560,
    the larger mammograms are cut to become 3328 x 2560 (default image crop shape)
    """

    def __init__(self,
                 image_crop_shape: Tuple[int, int],
                 orientation: str):
        """
        __init__ method: run one when instantiating the object

        :param image_crop_shape: image crop shape
        :param orientation: orientation image [L, R]
        :param debug: debug option
        """

        self.image_crop_shape = image_crop_shape

        self.orientation = orientation

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        if sample['image'].shape == self.image_crop_shape:
            image_crop = sample['image']
            image_w48m14_crop = sample['image']
            image_mask_crop = sample['image_mask']
            image_mask_w48m14_crop = sample['image_mask']
            annotation_crop = sample['annotation']
            annotation_w48m14_crop = sample['annotation_w48m14']
        else:
            # get min and max coordinates of annotation in x and y (for annotation and annotation-w48m14)
            annotation_x_min, annotation_x_max, annotation_y_min, annotation_y_max = min_max_annotation(annotation=sample['annotation'])
            annotation_w48m14_x_min, annotation_w48m14_x_max, annotation_w48m14_y_min, annotation_w48m14_y_max = min_max_annotation(annotation=sample['annotation_w48m14'])

            # get cutting point (on image with annotation)
            x_min_cut, x_max_cut, y_min_cut, y_max_cut = cutting_point(image_orientation=self.orientation,
                                                                       output_shape=self.image_crop_shape,
                                                                       input_shape=sample['image'].shape,
                                                                       x_constraints=(annotation_x_min, annotation_x_max),
                                                                       y_constraints=(annotation_y_min, annotation_y_max))

            # get cutting point (on image with annotation-w48m14)
            x_w48m14_min_cut, x_w48m14_max_cut, y_w48m14_min_cut, y_w48m14_max_cut = cutting_point(image_orientation=self.orientation,
                                                                                                   output_shape=self.image_crop_shape,
                                                                                                   input_shape=sample['image'].shape,
                                                                                                   x_constraints=(annotation_w48m14_x_min, annotation_w48m14_x_max),
                                                                                                   y_constraints=(annotation_w48m14_y_min, annotation_w48m14_y_max))

            # crop image
            image_crop = sample['image'][y_min_cut:y_max_cut, x_min_cut:x_max_cut]

            # crop image-w48m14
            # NOTE: annotations-w48m14 being different, the image crop must take this into account (one pixel gap)
            image_w48m14_crop = sample['image'][y_w48m14_min_cut:y_w48m14_max_cut, x_w48m14_min_cut:x_w48m14_max_cut]

            # crop image mask
            image_mask_crop = sample['image_mask'][y_min_cut:y_max_cut, x_min_cut:x_max_cut]

            # crop image-w48m14 mask
            image_mask_w48m14_crop = sample['image_mask'][y_w48m14_min_cut:y_w48m14_max_cut, x_w48m14_min_cut:x_w48m14_max_cut]

            # crop annotation
            annotation_crop = crop_annotation(annotation=sample['annotation'],
                                              x_crop=x_min_cut,
                                              y_crop=y_min_cut)

            # crop annotation-w48m14
            annotation_w48m14_crop = crop_annotation(annotation=sample['annotation_w48m14'],
                                                     x_crop=x_w48m14_min_cut,
                                                     y_crop=y_w48m14_min_cut)

        sample = {'filename': sample['filename'],
                  'image': image_crop,
                  'image_w48m14': image_w48m14_crop,
                  'image_mask': image_mask_crop,
                  'image_mask_w48m14': image_mask_w48m14_crop,
                  'annotation': annotation_crop,
                  'annotation_w48m14': annotation_w48m14_crop
                  }

        return sample


def min_max_annotation(annotation: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get the minimum and maximum of coordinates of annotation along x and y axes

    :param annotation: annotation
    :return: (x,y) min and (x,y) max of coords annotations
    """

    # init
    annotation_x = []
    annotation_y = []

    for line in annotation:
        x = int(line[1])
        y = int(line[2])

        annotation_x.append(x)
        annotation_y.append(y)

    if len(annotation_x) == 0 or len(annotation_y) == 0:
        x_min = x_max = y_min = y_max = -1
    else:
        x_min = min(annotation_x)
        x_max = max(annotation_x)
        y_min = min(annotation_y)
        y_max = max(annotation_y)

    return x_min, x_max, y_min, y_max


def cutting_point(image_orientation: str,
                  output_shape: Tuple[int, int],
                  input_shape: np.ndarray,
                  x_constraints: Tuple[int, int],
                  y_constraints: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Get image crop points based on constraints and image input and output dimensions

    :param image_orientation: image orientation [L, R]
    :param output_shape: output image shape
    :param input_shape: input image shape
    :param x_constraints: coords x constraints
    :param y_constraints: coords y constraints
    :return: (x,y) min and (x,y) max of cut
    """

    x_min, x_max = x_constraints
    y_min, y_max = y_constraints

    if x_max - x_min + 1 > output_shape[1] or y_max - y_min + 1 > output_shape[0]:
        str_err = "\nError: output_shape not enough for x or y constraints"
        sys.exit(str_err)

    if image_orientation == "L":
        x_min_cut = 0
        x_max_cut = output_shape[1]
    else:
        x_min_cut = input_shape[1] - output_shape[1]
        x_max_cut = input_shape[1]

    y_min_cut = int(input_shape[0] / 2) - int(output_shape[0] / 2)
    y_max_cut = y_min_cut + output_shape[0]

    if -1 not in y_constraints:

        delta_y_min = y_min - y_min_cut
        delta_y_max = y_max_cut - y_max
        delta_y_mean = int((delta_y_min + delta_y_max) / 2)

        if delta_y_min < delta_y_mean:
            y_shift = delta_y_mean - delta_y_min
            if y_min_cut - y_shift < 0:
                y_shift = y_min_cut
            y_min_cut -= y_shift
            y_max_cut -= y_shift
        elif delta_y_max < delta_y_mean:
            y_shift = delta_y_mean - delta_y_max
            if y_max_cut + y_shift > input_shape[0]:
                y_shift = input_shape[0] - y_max_cut
            y_min_cut += y_shift
            y_max_cut += y_shift

    return x_min_cut, x_max_cut, y_min_cut, y_max_cut


def crop_annotation(annotation: np.ndarray,
                    x_crop: int,
                    y_crop: int):
    """
    Crop annotation coordinates based on cutting point provided

    :param annotation: annotation
    :param x_crop: coords x crop
    :param y_crop: coords y crop
    :return: annotation cropped
    """

    annotation_crop = annotation.copy()

    if len(annotation) > 0:
        annotation_crop[:, 1] = annotation[:, 1] - x_crop
        annotation_crop[:, 2] = annotation[:, 2] - y_crop

    return annotation_crop
