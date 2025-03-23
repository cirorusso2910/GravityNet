import cv2
import numpy as np
import torch


def save_heatmap(heatmap: np.ndarray,
                 output_path: str,
                 scale_factor: float = 1.0):
    """
    Save Heatmap

    :param heatmap: heatmap
    :param output_path: output path
    :param scale_factor: scaling factor for aspect ratio
    """

    # convert the heatmap to numpy and normalize between 0 and 255
    # heatmap_np = heatmap.cpu().detach().numpy()
    heatmap_np = np.uint8(255 * (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()))

    # resize the heatmap based on the scale factor
    height, width = heatmap_np.shape
    new_size = (int(width * scale_factor), int(height * scale_factor))
    heatmap_resized = cv2.resize(heatmap_np, new_size, interpolation=cv2.INTER_LINEAR)

    # apply a colormap for visualization
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # save image
    cv2.imwrite(output_path, heatmap_color)
