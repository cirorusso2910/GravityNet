import sys
from typing import Tuple, Union

from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights

from net.model.backbone.ResNet.MyResNet101 import MyResNet101
from net.model.backbone.ResNet.MyResNet152 import MyResNet152
from net.model.backbone.ResNet.MyResNet18 import MyResNet18
from net.model.backbone.ResNet.MyResNet34 import MyResNet34
from net.model.backbone.ResNet.MyResNet50 import MyResNet50
from net.utility.msg.msg_error import msg_error


def MyResNet_models(resnet: int,
                    pretrained: bool) -> Tuple[Union[MyResNet18, MyResNet34, MyResNet50, MyResNet101, MyResNet152], int]:
    """
    Get ResNet models

    :param resnet: ResNet [18, 34, 50, 101, 152]
    :param pretrained: pretrained flag
    :return: ResNet model,
             num features
    """

    # --------- #
    # ResNet-18 #
    # --------- #
    if resnet == 18:
        ResNet_model = MyResNet18()  # MyResNet18 model
        if pretrained:
            # ResNet_model.load_state_dict(models.resnet18(pretrained=True).state_dict())
            ResNet_model.load_state_dict(models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict())
        num_features = 512  # num features out layer 4

    # --------- #
    # ResNet-34 #
    # --------- #
    elif resnet == 34:
        ResNet_model = MyResNet34()  # MyResNet34 model
        if pretrained:
            # ResNet_model.load_state_dict(models.resnet34(pretrained=True).state_dict())
            ResNet_model.load_state_dict(models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict())
        num_features = 512  # num features out layer 4

    # --------- #
    # ResNet-50 #
    # --------- #
    elif resnet == 50:
        ResNet_model = MyResNet50()  # MyResNet50 model
        if pretrained:
            # ResNet_model.load_state_dict(models.resnet50(pretrained=True).state_dict())
            ResNet_model.load_state_dict(models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).state_dict())
        num_features = 2048  # num features out layer 4

    # ---------- #
    # ResNet-101 #
    # ---------- #
    elif resnet == 101:
        ResNet_model = MyResNet101()  # MyResNet101 model
        if pretrained:
            # ResNet_model.load_state_dict(models.resnet101(pretrained=True).state_dict())
            ResNet_model.load_state_dict(models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1).state_dict())
        num_features = 2048  # num features out layer 4

    # ---------- #
    # ResNet-152 #
    # ---------- #
    elif resnet == 152:
        ResNet_model = MyResNet152()  # MyResNet152 model
        if pretrained:
            # ResNet_model.load_state_dict(models.resnet152(pretrained=True).state_dict())
            ResNet_model.load_state_dict(models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1).state_dict())
        num_features = 2048  # num features out layer 4

    else:
        str_err = msg_error(file=__file__,
                            variable=resnet,
                            type_variable="ResNet",
                            choices="[18, 34, 50, 101, 152]")
        sys.exit(str_err)

    return ResNet_model, num_features
