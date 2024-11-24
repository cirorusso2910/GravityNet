from torchvision.models.swin_transformer import SwinTransformer


class MySwinB(SwinTransformer):
    """
    My Swin-B
    """

    def __init__(self):

        super(MySwinB, self).__init__(patch_size=[4, 4],
                                      embed_dim=128,
                                      depths=[2, 2, 18, 2],
                                      num_heads=[4, 8, 16, 32],
                                      window_size=[7, 7],
                                      stochastic_depth_prob=0.5)

    def forward(self, x):

        x = self.features(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # permute from B x H_FM x W_FM x F to B x F x H_FM x W_FM

        return x
