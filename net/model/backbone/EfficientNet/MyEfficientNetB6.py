from functools import partial

from torch import nn
from torchvision.models import EfficientNet
from torchvision.models.efficientnet import _efficientnet_conf


class MyEfficientNetB6(EfficientNet):
    """
    My EfficientNet-B6
    """

    def __init__(self):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b6", width_mult=1.8, depth_mult=2.6)

        super(MyEfficientNetB6, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                               dropout=0.5,
                                               last_channel=last_channel,
                                               norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01))

    def forward(self, x):

        x = self.features(x)

        return x
