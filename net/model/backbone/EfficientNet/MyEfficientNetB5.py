from functools import partial

from torch import nn
from torchvision.models import EfficientNet
from torchvision.models.efficientnet import _efficientnet_conf


class MyEfficientNetB5(EfficientNet):
    """
    My EfficientNet-B5
    """

    def __init__(self):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b5", width_mult=1.6, depth_mult=2.2)

        super(MyEfficientNetB5, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                               dropout=0.4,
                                               last_channel=last_channel,
                                               norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),)

    def forward(self, x):

        x = self.features(x)

        return x
