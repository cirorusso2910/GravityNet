from torchvision.models import EfficientNet
from torchvision.models.efficientnet import _efficientnet_conf


class MyEfficientNetB3(EfficientNet):
    """
    My EfficientNet-B3
    """

    def __init__(self):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b3", width_mult=1.2, depth_mult=1.4)

        super(MyEfficientNetB3, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                               dropout=0.3,
                                               last_channel=last_channel)

    def forward(self, x):

        x = self.features(x)

        return x
