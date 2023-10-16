import sys
from typing import Tuple, Union

from torchvision import models
from torchvision.models import DenseNet121_Weights, DenseNet169_Weights, DenseNet161_Weights, DenseNet201_Weights

from net.model.backbone.DenseNet.MyDenseNet121 import MyDenseNet121
from net.model.backbone.DenseNet.MyDenseNet161 import MyDenseNet161
from net.model.backbone.DenseNet.MyDenseNet169 import MyDenseNet169
from net.model.backbone.DenseNet.MyDenseNet201 import MyDenseNet201
from net.utility.msg.msg_error import msg_error


def MyDenseNet_models(densenet: int,
                      pretrained: bool) -> Tuple[Union[MyDenseNet121, MyDenseNet161, MyDenseNet169, MyDenseNet201], int]:
    """
    Get DenseNet models

    :param densenet: DenseNet [121, 161, 169, 201]
    :param pretrained: pretrained flag
    :return: DenseNet model,
             num features
    """

    # ------------ #
    # DenseNet-121 #
    # ------------ #
    if densenet == 121:
        DenseNet_model = MyDenseNet121()  # MyDenseNet121 model
        if pretrained:
            DenseNet_model.load_state_dict(models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).state_dict())
        num_features = 1024  # num features out

    # ------------ #
    # DenseNet-161 #
    # ------------ #
    elif densenet == 161:
        DenseNet_model = MyDenseNet161()  # MyDenseNet161 model
        if pretrained:
            DenseNet_model.load_state_dict(models.densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1).state_dict())
        num_features = 2208  # num features out

    # ------------ #
    # DenseNet-169 #
    # ------------ #
    elif densenet == 169:
        DenseNet_model = MyDenseNet169()  # MyDenseNet169 model
        if pretrained:
            DenseNet_model.load_state_dict(models.densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1).state_dict())
        num_features = 1664  # num features out

    # ------------ #
    # DenseNet-201 #
    # ------------ #
    elif densenet == 201:
        DenseNet_model = MyDenseNet201()  # MyDenseNet201 model
        if pretrained:
            DenseNet_model.load_state_dict(models.densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1).state_dict())
        num_features = 1920  # num features out

    else:
        str_err = msg_error(file=__file__,
                            variable=densenet,
                            type_variable="DenseNet",
                            choices="[121, 161, 169, 201]")
        sys.exit(str_err)

    return DenseNet_model, num_features

