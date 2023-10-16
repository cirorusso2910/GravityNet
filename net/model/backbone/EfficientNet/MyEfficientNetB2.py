from torchvision.models import EfficientNet
from torchvision.models.efficientnet import _efficientnet_conf


class MyEfficientNetB2(EfficientNet):
    """
    My EfficientNet-B2
    """

    def __init__(self):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b2", width_mult=1.1, depth_mult=1.2)

        super(MyEfficientNetB2, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                               dropout=0.3,
                                               last_channel=last_channel)

    def forward(self, x):

        x = self.features(x)

        return x
