import torch.nn.functional as F

from torchvision.models import DenseNet


class MyDenseNet161(DenseNet):
    """
    My DenseNet-161
    """

    def __init__(self):
        super(MyDenseNet161, self).__init__(growth_rate=48,
                                            block_config=(6, 12, 36, 24),
                                            num_init_features=96)

    def forward(self, x):

        features = self.features(x)

        out = F.relu(features, inplace=True)

        return out
