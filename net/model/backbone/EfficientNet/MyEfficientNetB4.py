from torchvision.models import EfficientNet
from torchvision.models.efficientnet import _efficientnet_conf


class MyEfficientNetB4(EfficientNet):
    """
    My EfficientNet-B4
    """

    def __init__(self):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b4", width_mult=1.4, depth_mult=1.8)

        super(MyEfficientNetB4, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                               dropout=0.4,
                                               last_channel=last_channel)

    def forward(self, x):

        x = self.features(x)

        return x
