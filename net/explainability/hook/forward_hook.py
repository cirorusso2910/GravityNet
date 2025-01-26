from typing import Tuple

import torch
from torch import nn


def forward_hook(module: nn.Module,
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

    # define global variable
    global activations

    print("Forward hook is running...")
    activations = output  # capture the activations for use in Grad-CAM
    print("Activations Size --> ", activations.size())

    # suppress PyCharm warning for unused variables
    _ = module
    _ = input

