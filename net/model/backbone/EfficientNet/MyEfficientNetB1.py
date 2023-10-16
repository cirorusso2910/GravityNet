from torchvision.models import EfficientNet
from torchvision.models.efficientnet import _efficientnet_conf


class MyEfficientNetB1(EfficientNet):
    """
    My EfficientNet-B1
    """

    def __init__(self):
        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b1", width_mult=1.0, depth_mult=1.1)

        super(MyEfficientNetB1, self).__init__(inverted_residual_setting=inverted_residual_setting,
                                               dropout=0.2,
                                               last_channel=last_channel)

    def forward(self, x):

        x = self.features(x)

        return x
