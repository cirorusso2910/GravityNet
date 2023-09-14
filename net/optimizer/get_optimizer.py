import argparse
import sys

from torch.nn import Parameter
from torch.optim import Adam, SGD
from typing import Union, Iterator

from net.utility.msg.msg_error import msg_error


def get_optimizer(net_parameters: Iterator[Parameter],
                  parser: argparse.Namespace) -> Union[Adam, SGD]:
    """
    Get optimizer

    :param net_parameters: net parameters
    :param parser: parser of parameters-parsing
    :return: optimizer
    """

    # --------------- #
    # OPTIMIZER: ADAM #
    # --------------- #
    if parser.optimizer == 'Adam':
        optimizer = Adam(params=net_parameters,
                         lr=parser.learning_rate)

    # -------------- #
    # OPTIMIZER: SGD #
    # -------------- #
    elif parser.optimizer == 'SGD':
        optimizer = SGD(net_parameters,
                        lr=parser.learning_rate,
                        momentum=parser.lr_momentum)

    else:
        str_err = msg_error(file=__file__,
                            variable=parser.optimizer,
                            type_variable='optimizer',
                            choices='[Adam, SGD]')
        sys.exit(str_err)

    return optimizer
