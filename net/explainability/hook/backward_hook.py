from typing import Tuple

from torch import nn


def backward_hook(module: nn.Module,
                  grad_input: Tuple,
                  grad_output: Tuple):
    """
    Backward hook function to capture gradients during the backward pass.

    This function is executed when backpropagation reaches the layer where the hook is registered.
    It captures the gradients of the loss with respect to the activations of the hooked layer.

    :param module: module to which the hook is attached
    :param grad_input: gradients with respect to the inputs of the layer
    :param grad_output: gradients with respect to the outputs of the layer
    """

    # define global variable
    global gradients

    print('Backward hook running...')
    gradients = grad_output[0]
    print(f'Gradients size: {gradients[0].size()}')

    # suppress PyCharm warning for unused variables
    _ = module
    _ = grad_input
