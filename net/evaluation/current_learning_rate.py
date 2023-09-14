import argparse
import sys

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts
from typing import Union

from net.utility.msg.msg_error import msg_error


def current_learning_rate(scheduler: Union[ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts],
                          optimizer: Union[Adam, SGD],
                          parser: argparse.Namespace) -> float:
    """
    Get current learning rate according to scheduler and optimizer type

    :param scheduler: scheduler
    :param optimizer: optimizer
    :param parser: parser of parameters-parsing
    :return: current learning rate
    """

    if parser.scheduler == 'ReduceLROnPlateau':
        learning_rate = my_get_last_lr(optimizer=optimizer)

    elif parser.scheduler == 'StepLR':
        learning_rate = scheduler.get_last_lr()[0]

    elif parser.scheduler == 'CosineAnnealing':
        learning_rate = scheduler.get_last_lr()[0]

    else:
        str_err = msg_error(file=__file__,
                            variable=parser.scheduler,
                            type_variable='scheduler',
                            choices='[ReduceLROnPlateau, StepLR, CosineAnnealing]')
        sys.exit(str_err)

    return learning_rate


def my_get_last_lr(optimizer: Union[Adam, SGD]) -> float:
    """
    Get last Learning Rate
    :param optimizer: optimizer
    :return: last learning rate
    """

    for param_group in optimizer.param_groups:
        return param_group['lr']
