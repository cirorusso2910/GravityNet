from functools import partial

from torch import nn
from torchvision.models import EfficientNet
from torchvision.models.efficientnet import _efficientnet_conf


class MyEfficientNetV2L(EfficientNet):
    """
    My EfficientNetV2-L
    """

    def __init__(self):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")

        super(MyEfficientNetV2L, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                                dropout=0.4,
                                                last_channel=last_channel,
                                                norm_layer=partial(nn.BatchNorm2d, eps=1e-03))

    def forward(self, x):
        
        x = self.features(x)

        return x
