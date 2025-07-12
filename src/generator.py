import torch
import torch.nn as nn

'''
Class for Generator component for DCGAN (Deep Convolutional Adverserial Network)
'''
class Generator(nn.Module):
    def __init__(self, noise_dim=100, image_channels=3, feature_maps=64):
        super(Generator, self).__init__()

        '''
        Generator generates image from noise. That is why we use ConvTranspose2D as its internal layers, and output an image at the end.
        '''
        self.net = nn.Sequential(
            # Input: N * noise_dim * 1 * 1
            nn.ConvTranspose2d(noise_dim, feature_maps*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps*8),
            nn.ReLU(True),
            # Shape: (N, (feature_maps * 8), 4, 4): (number of images, feature_maps*8, [(I - 1)*S] - 2*P + K, [(I - 1)*S] - 2*P + K --- I = 1

            nn.ConvTranspose2d(feature_maps*8, feature_maps*4, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(feature_maps*4),
            nn.ReLU(True),
            # Shape: N * (feature_maps * 4) * 8 * 8: (number of images, feature_maps*8, [(I - 1)*S] - 2*P + K, [(I - 1)*S] - 2*P + K --- I = 4

            nn.ConvTranspose2d(feature_maps*4, feature_maps*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps*2),
            nn.ReLU(True),
            # Shape: N * (feature_maps * 2) * 16 * 16: (number of images, feature_maps*4, [(I - 1)*S] - 2*P + K, [(I - 1)*S] - 2*P + K --- I = 4

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # Shape: N x (feature_maps) x 32 x 32: (number of images, feature_maps*4, [(I - 1)*S] - 2*P + K, [(I - 1)*S] - 2*P + K --- I = 16

            nn.ConvTranspose2d(feature_maps, feature_maps // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps // 2),
            nn.ReLU(True),
            # Shape: N x (feature_maps//2) x 64 x 64: (number of images, feature_maps*4, [(I - 1)*S] - 2*P + K, [(I - 1)*S] - 2*P + K --- I = 32

            nn.ConvTranspose2d(feature_maps, feature_maps // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps // 2),
            nn.ReLU(True),
            # Shape: N x (feature_maps//2) x 128 x 128: (number of images, feature_maps*4, [(I - 1)*S] - 2*P + K, [(I - 1)*S] - 2*P + K --- I = 64

            nn.ConvTranspose2d(feature_maps // 4, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Final Output: N x 3 x 256 x 256: (number of images, feature_maps*4, [(I - 1)*S] - 2*P + K, [(I - 1)*S] - 2*P + K --- I = 128
        )
    
    '''
    Produce output from the network.
    '''
    def forward(self, x):
        return self.net(x)

'''
We want to be able to test the generator in case of any error.
'''
if __name__ == "__main__":
    noise_dim = 100
    G = Generator(noise_dim=noise_dim)
    noise = torch.randn(8, noise_dim, 1, 1)  # 8 images in batch
    fake_images = G(noise)
    print(fake_images.shape)  # Should print: torch.Size([8, 3, 128, 128])