import sys
from typing import Tuple, Union

from torchvision import models
from torchvision.models import ResNeXt50_32X4D_Weights, ResNeXt101_32X8D_Weights, ResNeXt101_64X4D_Weights

from net.model.backbone.ResNeXt.MyResNeXt101_32x8d import MyResNeXt101_32x8d
from net.model.backbone.ResNeXt.MyResNeXt101_64x4d import MyResNeXt101_64x4d
from net.model.backbone.ResNeXt.MyResNeXt50_32x4d import MyResNeXt50_32x4d
from net.utility.msg.msg_error import msg_error


def MyResNeXt_models(resnext: str,
                     pretrained: bool) -> Tuple[Union[MyResNeXt50_32x4d, MyResNeXt101_32x8d, MyResNeXt101_64x4d], int]:
    """
    Get ResNeXt models

    :param resnext: ResNeXt [50_32x4d, 101_32x8d, 101_64x4d]
    :param pretrained: pretrained flag
    :return: ResNeXt model,
             num features
    """

    # ---------------- #
    # ResNeXt-50_32x4d #
    # ---------------- #
    if resnext == "50_32x4d":
        ResNeXt_model = MyResNeXt50_32x4d()  # MyResNeXt50_32x4d model
        if pretrained:
            ResNeXt_model.load_state_dict(models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1).state_dict())
        num_features = 2048  # num features out layer 4

    # ----------------- #
    # ResNeXt-101_32x8d #
    # ----------------- #
    elif resnext == "101_32x8d":
        ResNeXt_model = MyResNeXt101_32x8d()  # MyResNeXt101_32x8d model
        if pretrained:
            ResNeXt_model.load_state_dict(models.resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V1).state_dict())
        num_features = 2048  # num features out layer 4

    # ----------------- #
    # ResNeXt-101_64x4d #
    # ----------------- #
    elif resnext == "101_64x4d":
        ResNeXt_model = MyResNeXt101_64x4d()  # MyResNeXt101_32x8d model
        if pretrained:
            # ResNet_model.load_state_dict(models.resnet18(pretrained=True).state_dict())
            ResNeXt_model.load_state_dict(models.resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1).state_dict())
        num_features = 2048  # num features out layer 4

    else:
        str_err = msg_error(file=__file__,
                            variable=resnext,
                            type_variable="ResNeXt",
                            choices="[50_32x4d, 101_32x8d, 101_64x4d]")
        sys.exit(str_err)

    return ResNeXt_model, num_features
