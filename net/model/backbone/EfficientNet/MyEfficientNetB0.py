from torchvision.models import EfficientNet
from torchvision.models.efficientnet import _efficientnet_conf


class MyEfficientNetB0(EfficientNet):
    """
    My EfficientNet-B0
    """

    def __init__(self):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)

        super(MyEfficientNetB0, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                               dropout=0.2,
                                               last_channel=last_channel)

    def forward(self, x):

        x = self.features(x)

        return x
