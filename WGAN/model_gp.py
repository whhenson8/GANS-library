## Architecture:
## Generator: Fractional Strided Conv (FSC) -> FSC + ReLU -> FSC + ReLU -> FSC + ReLU 
##                                          -> FSC + ReLU -> Tanh activation
## Discriminator: Conv -> LReLU -> FSC + BatchNorm + LReLU -> FSC + BatchNorm + LReLU
##                              -> FSC + BatchNorm + LReLU -> Conv -> sigmoid activation

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_channels, image_channels, features_gen):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(noise_channels, features_gen * 16, 4, 1, 0),  # img: 4x4
            self._block(features_gen * 16, features_gen * 8, 4, 2, 1),  # img: 8x8
            self._block(features_gen * 8, features_gen * 4, 4, 2, 1),  # img: 16x16
            self._block(features_gen * 4, features_gen * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_gen*2, image_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),
            self._block(features_d*2, features_d*4, 4, 2, 1),
            self._block(features_d*4, features_d*8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine = True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

def initialize_weights(model):
    # Initializes weights according to the original DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)       # all initialised in the same way.


def test():
    N, in_channels, H, W = 8, 3, 64, 64 # num_ims, in_channels, height, width, as usual.
    noise_dim = 100                     # noise channel allows generator to introduce variability of generated samples. Discriminator reshapes to give meaningful input
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")


if __name__ == "__main__":
    test()