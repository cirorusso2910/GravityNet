import cv2
import numpy as np
import torch

from typing import Tuple

from net.dataset.utility.viewable_image import viewable_image


def save_image_overlay(image: torch.Tensor,
                       heatmap: np.ndarray,
                       size: Tuple[int, int],
                       output_path: str):
    """
    Save overlay image with a Grad-CAM heatmap on top of an input image.

    :param image: original input image
    :param heatmap: heatmap generated by the Grad-CAM algorithm
    :param size: original input image size
    :param output_path: path to save the overlay image
    """

    # convert image to numpy format and BGR for OpenCV
    image_np = image.permute((1, 2, 0))  # permute
    image_np = image_np.cpu().detach().numpy()
    image_np = viewable_image(image=image_np)

    # convert heatmap to numpy format
    # heatmap_np = heatmap.cpu().detach().numpy()
    heatmap_np = np.uint8(255 * (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()))
    # resize heatmap to match original image size
    heatmap_resize = cv2.resize(heatmap_np, size, interpolation=cv2.INTER_LINEAR)
    # apply color map to the heatmap
    heatmap_color = cv2.applyColorMap(heatmap_resize, cv2.COLORMAP_JET)

    # overlay image
    overlay_image = cv2.addWeighted(image_np, 0.7, heatmap_color, 0.3, 0)

    # save the resulting overlay image
    cv2.imwrite(output_path, overlay_image)
