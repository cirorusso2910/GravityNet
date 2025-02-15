import torch

from torch import nn


class ClassificationModel(nn.Module):
    """
    Classification SubNet
    """

    def __init__(self,
                 num_features_in: int,
                 num_gravity_points_feature_map: int,
                 num_classes: int = 2,
                 feature_size: int = 256):
        """
        __init__ method: run one when instantiating the object

        :param num_features_in: num features input
        :param num_gravity_points_feature_map: num gravity points for feature map
        :param num_classes: num classes (default: 2)
        :param feature_size: feature size (default: 256)
        """

        super(ClassificationModel, self).__init__()

        # num classes
        self.num_classes = num_classes
        # num gravity points
        self.num_gravity_points_feature_map = num_gravity_points_feature_map

        # convolutional layer 3x3 (256)
        self.conv1 = nn.Conv2d(in_channels=num_features_in, out_channels=feature_size, kernel_size=3, padding=1)
        # batch normalization
        self.bn1 = nn.BatchNorm2d(num_features=feature_size)

        # convolutional layer 3x3 (256)
        self.conv2 = nn.Conv2d(in_channels=feature_size, out_channels=feature_size, kernel_size=3, padding=1)
        # batch normalization
        self.bn2 = nn.BatchNorm2d(num_features=feature_size)

        # convolutional layer 3x3 (256)
        self.conv3 = nn.Conv2d(in_channels=feature_size, out_channels=feature_size, kernel_size=3, padding=1)
        # batch normalization
        self.bn3 = nn.BatchNorm2d(num_features=feature_size)

        # convolutional layer 3x3 (256)
        self.conv4 = nn.Conv2d(in_channels=feature_size, out_channels=feature_size, kernel_size=3, padding=1)
        # batch normalization
        self.bn4 = nn.BatchNorm2d(num_features=feature_size)

        # Xavier initialization
        torch.manual_seed(seed=0)
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)

        # activation: ReLU
        self.act = nn.ReLU()

        # activation: Sigmoid
        self.output_act = nn.Sigmoid()

        # convolutional layer 3x3 (A*K)
        self.output = nn.Conv2d(in_channels=feature_size, out_channels=num_gravity_points_feature_map * num_classes, kernel_size=3, padding=1)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        forward method: directly call a method in the class when an instance name is called

        :param x: input
        :return: classification output
        """

        # pass data through activation function, output normalized, in conv1
        out = self.act(self.bn1(self.conv1(x)))

        # pass data through activation function, output normalized, in conv2
        out = self.act(self.bn2(self.conv2(out)))

        # pass data through activation function, output normalized, in conv3
        out = self.act(self.bn3(self.conv3(out)))

        # pass data through activation function, output normalized, in conv4
        out = self.act(self.bn4(self.conv4(out)))

        # pass data through activation function and output conv
        out = self.output_act(self.output(out))

        # out is B x C x W x H, with C = num_classes + num_gravity_points
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_gravity_points_feature_map, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)
