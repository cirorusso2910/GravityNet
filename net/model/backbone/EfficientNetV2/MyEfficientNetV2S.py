from functools import partial

from torch import nn
from torchvision.models import EfficientNet
from torchvision.models.efficientnet import _efficientnet_conf


class MyEfficientNetV2S(EfficientNet):
    """
    My EfficientNetV2-S
    """

    def __init__(self):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s", width_mult=2.0, depth_mult=3.1)

        super(MyEfficientNetV2S, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                                dropout=0.2,
                                                last_channel=last_channel,
                                                norm_layer=partial(nn.BatchNorm2d, eps=1e-03))

    def forward(self, x):

        x = self.features(x)

        return x
