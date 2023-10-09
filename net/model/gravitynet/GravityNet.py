import torch
import torch.nn as nn

from typing import Tuple

from net.model.gravitynet.ClassificationSubNet import ClassificationModel
from net.model.gravitynet.RegressionSubNet import RegressionModel
from net.model.MyResNet_models import MyResNet_models


class GravityNet(nn.Module):
    """
    Gravity Network
    """

    def __init__(self,
                 backbone: str,
                 pretrained: bool,
                 num_gravity_points_feature_map: int):
        """
        __init__ method: run one when instantiating the object

        :param backbone: backbone
        :param pretrained: pretrained flag
        :param num_gravity_points_feature_map: num gravity points for feature map
        """

        super(GravityNet, self).__init__()

        # PreTrained (True/False)
        self.pretrained = pretrained

        # num gravity points in feature map (reference window)
        self.num_gravity_points_feature_map = num_gravity_points_feature_map

        # -------------- #
        # Backbone Model #
        # -------------- #
        # - ResNet
        if backbone.split('-')[0] == 'ResNet':

            # ResNet [18, 34, 50, 101, 152]
            resnet = int(backbone.split('-')[1])

            # ResNet Model
            self.backboneModel, self.num_features = MyResNet_models(resnet=resnet,
                                                                    pretrained=self.pretrained)
        # ----------------- #
        # Regression SubNet #
        # ----------------- #
        self.regressionModel = RegressionModel(num_features_in=self.num_features,
                                               num_gravity_points_feature_map=self.num_gravity_points_feature_map)

        # --------------------- #
        # Classification SubNet #
        # --------------------- #
        self.classificationModel = ClassificationModel(num_features_in=self.num_features,
                                                       num_classes=2,
                                                       num_gravity_points_feature_map=self.num_gravity_points_feature_map)

    def forward(self,
                image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        forward method: directly call a method in the class when an instance name is called

        :param image: image
        :return: classification subnet output,
                 regression subnet output
        """

        # Backbone
        backbone_output = self.backboneModel(image)  # backbone output shape: B x F x H_FM x W_FM

        # Regression SubNet
        regression_output = self.regressionModel(backbone_output)  # regression shape: B x A x 2

        # Classification SubNet
        classification_output = self.classificationModel(backbone_output)  # classification shape: B x A x 2

        return classification_output, regression_output
