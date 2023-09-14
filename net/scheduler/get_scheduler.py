import argparse
import sys

from typing import Union

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts

from net.utility.msg.msg_error import msg_error


def get_scheduler(optimizer: Union[Adam, SGD],
                  parser: argparse.Namespace) -> Union[ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts]:
    """
    Get scheduler

    :param optimizer: optimizer
    :param parser: parser of parameters-parsing
    :return: scheduler
    """

    # ----------------- #
    # ReduceLROnPlateau #
    # ----------------- #
    if parser.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                      patience=parser.lr_patience,
                                      verbose=True)

    # ------ #
    # StepLR #
    # ------ #
    elif parser.scheduler == 'StepLR':
        scheduler = StepLR(optimizer=optimizer,
                           step_size=parser.lr_step_size,
                           gamma=parser.lr_gamma)

    # --------------- #
    # CosineAnnealing #
    # --------------- #
    elif parser.scheduler == "CosineAnnealing":
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                T_0=parser.lr_T0)

    else:
        str_err = msg_error(file=__file__,
                            variable=parser.scheduler,
                            type_variable='scheduler',
                            choices='[ReduceLROnPlateau, StepLR, CosineAnnealing]')
        sys.exit(str_err)

    return scheduler
