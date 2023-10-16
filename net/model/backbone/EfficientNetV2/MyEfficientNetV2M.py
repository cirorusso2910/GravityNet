from functools import partial

from torch import nn
from torchvision.models import EfficientNet
from torchvision.models.efficientnet import _efficientnet_conf


class MyEfficientNetV2M(EfficientNet):
    """
    My EfficientNetV2-M
    """

    def __init__(self):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")

        super(MyEfficientNetV2M, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                                dropout=0.3,
                                                last_channel=last_channel,
                                                norm_layer=partial(nn.BatchNorm2d, eps=1e-03))

    def forward(self, x):
        
        x = self.features(x)

        return x
