import torch
from torch import nn


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator for paired conditional images."""

    def __init__(self, in_channels=6, base_channels=64):
        super().__init__()
        c = base_channels

        def block(cin, cout, stride=2, normalize=True):
            layers = [
                nn.Conv2d(cin, cout, kernel_size=4, stride=stride, padding=1, bias=not normalize)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        layers += block(in_channels, c, normalize=False)
        layers += block(c, c * 2)
        layers += block(c * 2, c * 4)
        layers += block(c * 4, c * 8, stride=1)
        layers.append(nn.Conv2d(c * 8, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, condition, image):
        return self.model(torch.cat([condition, image], dim=1))
