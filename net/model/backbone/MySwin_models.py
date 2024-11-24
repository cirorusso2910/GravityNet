import sys
from typing import Tuple, Union

from torchvision import models
from torchvision.models import Swin_T_Weights, Swin_S_Weights, Swin_B_Weights

from net.model.backbone.Swin.MySwinB import MySwinB
from net.model.backbone.Swin.MySwinS import MySwinS
from net.model.backbone.Swin.MySwinT import MySwinT
from net.utility.msg.msg_error import msg_error


def MySwin_models(swin: str,
                  pretrained: bool) -> Tuple[Union[MySwinT, MySwinS, MySwinB], int]:
    """
    Get Swin models

    :param swin: Swin [T, S, B]
    :param pretrained: pretrained flag
    :return: Swin model,
             num features
    """

    # ------ #
    # Swin-T #
    # ------ #
    if swin == 'T':
        Swin_model = MySwinT()  # MySwinT model
        if pretrained:
            Swin_model.load_state_dict(models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).state_dict())
        num_features = 768

    # ------ #
    # Swin-S #
    # ------ #
    elif swin == 'S':
        Swin_model = MySwinS()  # MySwinT model
        if pretrained:
            Swin_model.load_state_dict(models.swin_s(weights=Swin_S_Weights.IMAGENET1K_V1).state_dict())
        num_features = 768

    # ------ #
    # Swin-B #
    # ------ #
    elif swin == 'B':
        Swin_model = MySwinB()  # MySwinT model
        if pretrained:
            Swin_model.load_state_dict(models.swin_b(weights=Swin_B_Weights.IMAGENET1K_V1).state_dict())
        num_features = 1024

    else:
        str_err = msg_error(file=__file__,
                            variable=swin,
                            type_variable="Swin",
                            choices="[T, S, B]")
        sys.exit(str_err)

    return Swin_model, num_features
