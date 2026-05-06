import torch
from torch import nn


def _down_block(in_channels, out_channels, normalize=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize)
    ]
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _up_block(in_channels, out_channels, dropout=0.0):
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class GeneratorUNet(nn.Module):
    """U-Net generator used by Pix2Pix for 256x256 image translation."""

    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        c = base_channels
        self.down1 = _down_block(in_channels, c, normalize=False)
        self.down2 = _down_block(c, c * 2)
        self.down3 = _down_block(c * 2, c * 4)
        self.down4 = _down_block(c * 4, c * 8)
        self.down5 = _down_block(c * 8, c * 8)
        self.down6 = _down_block(c * 8, c * 8)
        self.down7 = _down_block(c * 8, c * 8)
        self.down8 = _down_block(c * 8, c * 8, normalize=False)

        self.up1 = _up_block(c * 8, c * 8, dropout=0.5)
        self.up2 = _up_block(c * 16, c * 8, dropout=0.5)
        self.up3 = _up_block(c * 16, c * 8, dropout=0.5)
        self.up4 = _up_block(c * 16, c * 8)
        self.up5 = _up_block(c * 16, c * 4)
        self.up6 = _up_block(c * 8, c * 2)
        self.up7 = _up_block(c * 4, c)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(c * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        return self.final(torch.cat([u7, d1], dim=1))
