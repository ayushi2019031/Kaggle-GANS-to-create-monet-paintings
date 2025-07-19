import torch
import torch.nn as nn

'''
A residual block is a building block used in ResNet which helps neural networks learn. 
Basically it learns the change it wants to apply in the input and then apply the change back in. 
This helps in gradient flow in backpropagation and reuse of low-level features.
'''
class ResidualBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # mirrors the edge pixels. 
            nn.Conv2d(features, features, kernel_size=3),
            nn.InstanceNorm2d(features), # while batch norm normalizes images with mean and std deviation ACROSS ENTIRE BATCH,
            # instance norm 2d normalizes image across per image, per channel.  
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3),
            nn.InstanceNorm2d(features),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=9):
        super().__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        ]

        # Downsampling - we shrink the image to smaller size, but increase the feature channels -
        # capturing information about more features. 
        in_features = 64
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, in_features*2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(in_features*2),
                nn.ReLU(True),
            ]
            in_features *= 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, in_features//2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(in_features//2),
                nn.ReLU(True),
            ]
            in_features //= 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
