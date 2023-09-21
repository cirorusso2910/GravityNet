import numpy as np
import torch

from net.dataset.utility.viewable_image import viewable_image


def image_tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    """
    Image conversion tensor to numpy for drawing output

    :param image: image
    :return: image converted
    """

    # convert tensor image in numpy and permute
    if torch.is_tensor(image):
        image = image.permute((1, 2, 0))  # permute 3xHxW -> HxWx3
        image = image.cpu().detach().numpy()  # to numpy
        image = viewable_image(image=image)  # viewable

    return image
