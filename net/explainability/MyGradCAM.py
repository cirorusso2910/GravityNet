from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class MyGradCAM:
    """
    My GradCAM
    """

    def __init__(self,
                 backbone: str):
        """
        __init__ method: run one when instantiating the object

        :param backbone: backbone model
        """

        self.gradients = None
        self.activations = None

        self.backbone = backbone

    def backward_hook(self,
                      module: nn.Module,
                      grad_input: Tuple,
                      grad_output: Tuple):
        """
        Backward hook function to capture gradients during the backward pass.

        This function is executed when backpropagation reaches the layer where the hook is registered.
        It captures the gradients of the loss with respect to the activations of the hooked layer.

        :param module: module (layer) to which the hook is attached
        :param grad_input: gradients with respect to the inputs of the layer
        :param grad_output: gradients with respect to the outputs of the layer
        """

        print("Backward hook is running...")

        # - ResNet model
        if 'ResNet' in self.backbone:
            self.gradients = grad_output  # capture the gradients for use in Grad-CAM
        # - Swin model
        elif 'Swin' in self.backbone:
            self.gradients = grad_output  # capture the gradients for use in Grad-CAM
            self.gradients = tuple(grad.permute(0, 3, 1, 2) for grad in self.gradients)  # permute gradients
        else:
            ValueError('Backward hook only supports: ResNet, Swin')

        print("Gradients Size --> ", self.gradients[0].size())

        # suppress PyCharm warning for unused variables
        _ = module
        _ = grad_input

    def forward_hook(self,
                     module: nn.Module,
                     input: Tuple,
                     output: torch.Tensor):
        """
        Forward hook function to capture activations during the forward pass.

        This function is executed during the forward pass, capturing the activations (feature maps)
        from the layer where the hook is registered.

        :param module: module (layer) to which the hook is attached
        :param input: input to the layer
        :param output: output (activations) from the layer
        """

        print("Forward hook is running...")

        # - ResNet model
        if 'ResNet' in self.backbone:
            self.activations = output  # capture the activations for use in Grad-CAM
        # - Swin model
        elif 'Swin' in self.backbone:
            self.activations = output.permute(0, 3, 1, 2)  # capture the activations for use in Grad-CAM
        else:
            ValueError('Backward hook only supports: ResNet, Swin')

        print("Activations Size --> ", self.activations.size())

        # suppress PyCharm warning for unused variables
        _ = module
        _ = input

    def heatmap(self) -> torch.Tensor:
        """
        Compute Grad-CAM (Gradient-weighted Class Activation Mapping).

        param gradients: gradients of the output with respect to the feature map of a layer
        param activations: activations (output feature map) from the forward pass of that layer
        :return: heatmap generated from the activations and gradients
        """

        # pool the gradients across the channels (global average pooling over width and height dimensions)
        pooled_gradients = torch.mean(self.gradients[0], dim=[0, 2, 3])

        # reshape pooled_gradients for broadcasting
        pooled_gradients = pooled_gradients.view(1, -1, 1, 1)

        # weight each channel in the activations by the corresponding pooled gradients (using broadcasting)
        activations = self.activations * pooled_gradients

        # normalize
        activations = F.normalize(activations)

        # compute the mean of the weighted activations across all channels
        heatmap = torch.mean(activations, dim=1).squeeze()

        # apply ReLU to remove negative values (keep only positive influences)
        heatmap = F.relu(heatmap)

        # normalize the heatmap to the range [0, 1] (optional, but useful for visualization)
        heatmap /= torch.max(heatmap)

        return heatmap
