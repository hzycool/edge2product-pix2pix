import torch
from torch import nn

from .discriminator_patchgan import PatchGANDiscriminator
from .generator_unet import GeneratorUNet
from .losses import GANLoss, l1_loss


class Pix2PixModel(nn.Module):
    """Small training wrapper around Generator, Discriminator, and Pix2Pix losses."""

    def __init__(self, lambda_l1=100.0, lr=2e-4, beta1=0.5, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.lambda_l1 = lambda_l1
        self.netG = GeneratorUNet().to(self.device)
        self.netD = PatchGANDiscriminator().to(self.device)
        self.gan_loss = GANLoss().to(self.device)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))

    def set_input(self, batch):
        self.real_A = batch["input"].to(self.device)
        self.real_B = batch["target"].to(self.device)

    def optimize_parameters(self):
        self.fake_B = self.netG(self.real_A)

        self.optimizer_D.zero_grad(set_to_none=True)
        pred_real = self.netD(self.real_A, self.real_B)
        loss_D_real = self.gan_loss(pred_real, True)
        pred_fake = self.netD(self.real_A, self.fake_B.detach())
        loss_D_fake = self.gan_loss(pred_fake, False)
        loss_D = 0.5 * (loss_D_real + loss_D_fake)
        loss_D.backward()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad(set_to_none=True)
        pred_fake_for_G = self.netD(self.real_A, self.fake_B)
        loss_G_GAN = self.gan_loss(pred_fake_for_G, True)
        loss_G_L1 = l1_loss(self.fake_B, self.real_B) * self.lambda_l1
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        self.optimizer_G.step()

        return {
            "G_loss": float(loss_G.detach().cpu()),
            "D_loss": float(loss_D.detach().cpu()),
            "GAN_loss": float(loss_G_GAN.detach().cpu()),
            "L1_loss": float(loss_G_L1.detach().cpu()),
        }
