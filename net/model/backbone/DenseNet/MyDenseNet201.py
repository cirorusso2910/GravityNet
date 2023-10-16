import torch.nn.functional as F

from torchvision.models import DenseNet


class MyDenseNet201(DenseNet):
    """
    My DenseNet-201
    """

    def __init__(self):
        super(MyDenseNet201, self).__init__(growth_rate=32,
                                            block_config=(6, 12, 48, 32),
                                            num_init_features=64)

    def forward(self, x):

        features = self.features(x)

        out = F.relu(features, inplace=True)

        return out
