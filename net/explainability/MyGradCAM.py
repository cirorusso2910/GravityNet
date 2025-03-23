from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from net.loss.GravityLoss import GravityLoss


class MyGradCAM:
    """
    My GradCAM
    """

    def __init__(self,
                 model: nn.Module,
                 criterion: Union[GravityLoss],
                 lambda_factor: int,
                 gravity_points: np.ndarray,
                 target_layer):
        """
        __init__ method: run one when instantiating the object

        :param model: model
        :param criterion: loss function
        :param lambda_factor: lambda factor (loss)
        :param gravity_points: gravity points configuration
        :param target_layer: target layer
        """

        # model
        self.model = model

        # criterion
        self.criterion = criterion
        self.lambda_factor = lambda_factor

        # gravity points configuration
        self.gravity_points = gravity_points

        # target layer
        self.target_layer = target_layer

        # gradients
        self.gradients = None

        # activations
        self.activations = None

        # Register hooks to capture gradients and activations
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """
        Hook function that captures the forward activations of the target layer
        'output' represents the feature maps produced by this layer when processing the input image

        these activations show what patterns the network detected in the input image
        """

        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """
        Hook function that captures the gradients flowing back through the target layer during backpropagation
        'grad_output' contains the gradients of the loss with respect to the outputs of this layer

        these gradients indicate how important each activation is for the target class prediction
        """

        self.gradients = grad_output[0].detach()

    def generate_heatmap(self,
                         image: torch.Tensor,
                         annotation: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Generate the activation heatmap

        :param image: image
        :param annotation: annotation
        :return: normalized activation map
        """

        # set the model to evaluation mode
        self.model.eval()

        # forward pass
        classifications, regressions = self.model(image)

        # compute loss
        classification_loss, regression_loss = self.criterion(images=image,
                                                              classifications=classifications,
                                                              regressions=regressions,
                                                              gravity_points=self.gravity_points,
                                                              annotations=annotation)

        # compute final loss
        loss = classification_loss + self.lambda_factor * regression_loss

        # reset parameters gradients of the model (net)
        self.model.zero_grad()

        # loss gradient backpropagation
        loss.backward()

        # calculate the weight coefficients
        gradients = self.gradients.mean(dim=(2, 3), keepdim=True)

        # multiply weights by activations
        cam = torch.sum(gradients * self.activations, dim=1).squeeze()

        # apply ReLU to the CAM
        cam = F.relu(cam)

        # normalize
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()
