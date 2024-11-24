from torchvision.models.swin_transformer import SwinTransformer


class MySwinT(SwinTransformer):
    """
    My Swin-T
    """

    def __init__(self):
        super(MySwinT, self).__init__(patch_size=[4, 4],
                                      embed_dim=96,
                                      depths=[2, 2, 6, 2],
                                      num_heads=[3, 6, 12, 24],
                                      window_size=[7, 7],
                                      stochastic_depth_prob=0.2)

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # permute from B x H_FM x W_FM x F to B x F x H_FM x W_FM

        return x
