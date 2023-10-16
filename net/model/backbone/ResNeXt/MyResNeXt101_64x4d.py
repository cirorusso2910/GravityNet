from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck


class MyResNeXt101_64x4d(ResNet):
    """
    My ResNeXt-101_64x4d
    """

    def __init__(self):
        super(MyResNeXt101_64x4d, self).__init__(block=Bottleneck,
                                                 layers=[3, 4, 23, 3],
                                                 groups=64,
                                                 width_per_group=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # Layer 1 (64)
        x = self.layer2(x)  # Layer 2 (128)
        x = self.layer3(x)  # Layer 3 (256)
        x = self.layer4(x)  # Layer 4 (512)

        return x
