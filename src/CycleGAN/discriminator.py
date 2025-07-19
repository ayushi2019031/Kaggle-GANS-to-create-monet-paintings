import torch
import torch.nn as nn

'''
Discrinimator architecture for cyclic GAN. 
'''
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def conv_block(in_features, out_features, normalize=True):
            layers = [nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1)]
            if normalize:
                '''
                Use instance norm over batch norm for generating images. 
                '''
                layers.append(nn.InstanceNorm2d(out_features))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 70x70 PatchGAN
        self.model = nn.Sequential(
            *conv_block(in_channels, 64, normalize=False),  # 256 → 128
            *conv_block(64, 128),                           # 128 → 64
            *conv_block(128, 256),                          # 64 → 32
            *conv_block(256, 512),                          # 32 → 16
            nn.Conv2d(512, 1, kernel_size=4, padding=1)     # 16 → ~14 (patch map)
        )

    def forward(self, x):
        return self.model(x)
