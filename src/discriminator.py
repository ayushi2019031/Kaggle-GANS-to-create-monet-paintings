import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminator network for GAN that classifies 128x128 RGB images as real or fake.
    Uses strided convolutions to progressively downsample the input.
    """
    def __init__(self, image_channels=3, feature_maps=64):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            # Input: (N, 3, 128, 128)
            nn.Conv2d(image_channels, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # (N, 64, 64, 64)

            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (N, 128, 32, 32)

            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (N, 256, 16, 16)

            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (N, 512, 8, 8)

            nn.Conv2d(feature_maps * 8, feature_maps * 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # (N, 1024, 4, 4)

            nn.Conv2d(feature_maps * 16, 1, kernel_size=4, stride=1, padding=0),
            # (N, 1, 1, 1)

            nn.Sigmoid()  # Output a probability
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1)  # Flatten to (N,)
    
'''
We want to be able to test the discriminator individually in case of any errors.
'''
if __name__ == "__main__":
    D = Discriminator()
    images = torch.randn(8, 3, 128, 128)  # batch of fake or real images
    out = D(images)
    print(out.shape)  # Should print: torch.Size([8])
