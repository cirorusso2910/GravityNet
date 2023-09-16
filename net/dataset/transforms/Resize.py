import cv2
import numpy as np
import os
import sys
import torch

from typing import Tuple

from torchvision.transforms import transforms, InterpolationMode

from net.utility.msg.msg_error import msg_error


class Resize(object):
    """
    Resize image and annotation according to size (H x W) and tool (PyTorch, OpenCV)
    """

    def __init__(self,
                 size: Tuple[int, int],
                 tool: str):
        """
        __init__ method: run one when instantiating the object

        :param size: image size
        :param tool: resize tool
        """

        self.size = size

        self.tool = tool

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # read sample
        image = sample['image']
        annotation = sample['annotation']
        image_mask = sample['image_mask']

        # using PyTorch as resize tool
        if self.tool == 'PyTorch':

            # transforms torchvision resize
            transforms_resize = transforms.Resize(self.size, interpolation=InterpolationMode.NEAREST, max_size=None, antialias=None)

            # resize image
            image = torch.from_numpy(image)  # numpy to tensor
            image = image.permute(2, 0, 1)  # permute [H, W, C] -> [C, H, W]
            image_resize = transforms_resize(image)
            image_resize = image_resize.cpu().detach().numpy()  # tensor to numpy
            image = image.cpu().detach().numpy()  # tensor to numpy

            # resize annotation
            x_scale = image_resize.shape[1] / image.shape[1]  # x scale resize
            y_scale = image_resize.shape[2] / image.shape[2]  # y scale resize
            annotation_resize = annotation.copy()
            annotation_resize[:, 0] = annotation[:, 0] * y_scale  # resize coord x
            annotation_resize[:, 1] = annotation[:, 1] * x_scale  # resize coord y
            annotation_resize[:, 2] = annotation[:, 2] * x_scale  # resize radius in x
            annotation_resize[:, 3] = annotation[:, 3] * y_scale  # resize radius in y

            # resize image mask
            image_mask = torch.from_numpy(image_mask)  # numpy to tensor
            image_mask = image_mask.permute(2, 0, 1)  # permute [H, W, C] -> [C, H, W]
            image_mask_resize = transforms_resize(image_mask)
            image_mask_resize = image_mask_resize.cpu().detach().numpy()[0, :, :]  # tensor to numpy
            image_mask = image_mask.cpu().detach().numpy()[0, :, :]  # tensor to numpy

        # using openCV as resize tool
        elif self.tool == 'openCV':

            # resize image
            image_resize_shape = (self.size[1], self.size[0])
            image_resize = cv2.resize(image, image_resize_shape, interpolation=cv2.INTER_AREA)
            image_resize = np.transpose(image_resize, (2, 0, 1))
            image = np.transpose(image, (2, 0, 1))

            # resize annotation
            x_scale = image_resize.shape[1]/image.shape[1]  # x scale resize
            y_scale = image_resize.shape[2]/image.shape[2]  # y scale resize
            annotation_resize = annotation.copy()
            annotation_resize[:, 0] = annotation[:, 0] * y_scale  # resize coord x
            annotation_resize[:, 1] = annotation[:, 1] * x_scale  # resize coord y
            annotation_resize[:, 2] = annotation[:, 2] * x_scale  # resize radius in x
            annotation_resize[:, 3] = annotation[:, 3] * y_scale  # resize radius in y

            # resize image mask
            image_mask_resize = cv2.resize(image_mask, image_resize_shape, interpolation=cv2.INTER_AREA)
            image_mask_resize = np.transpose(image_mask_resize, (2, 0, 1))
            image_mask_resize = image_mask_resize[0]

        else:
            str_err = msg_error(file=__file__,
                                variable=self.tool,
                                type_variable="Resize tool",
                                choices="[PyTorch, openCV]")
            sys.exit(str_err)

        sample = {'filename': sample['filename'],
                  'image': image_resize,
                  'image_mask': image_mask_resize,
                  'annotation': annotation_resize,
                  }

        return sample
