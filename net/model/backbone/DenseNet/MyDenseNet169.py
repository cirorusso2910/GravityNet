import torch.nn.functional as F

from torchvision.models import DenseNet


class MyDenseNet169(DenseNet):
    """
    My DenseNet-169
    """

    def __init__(self):
        super(MyDenseNet169, self).__init__(growth_rate=32,
                                            block_config=(6, 12, 32, 32),
                                            num_init_features=64)

    def forward(self, x):

        features = self.features(x)

        out = F.relu(features, inplace=True)

        return out
