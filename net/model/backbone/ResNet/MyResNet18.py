from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock


class MyResNet18(ResNet):
    """
    My ResNet-18
    """

    def __init__(self):

        super(MyResNet18, self).__init__(block=BasicBlock,
                                         layers=[2, 2, 2, 2])

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
