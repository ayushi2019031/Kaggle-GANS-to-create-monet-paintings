import random
import torch
import matplotlib.pyplot as plt

'''
queue of previously generated fake images. Instead of always feeding new 
fake images to the discriminator, we sometimes reuse older ones.
This helps the discriminator: avoid overfitting to the generator's latest outputs and
see a more diverse set of "fake" examples
'''
class ReplayBuffer():
    def __init__(self, max_size=50):
        """
        Stores previously generated images to stabilize GAN training.
        """
        assert max_size > 0, "Buffer size must be > 0"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, new_data):
        """
        new_data: Tensor of fake images from generator [B, C, H, W]
        Returns: Tensor of images for discriminator
        """
        result = []
        for image in new_data:
            image = torch.unsqueeze(image.data, 0)  # Add batch dim
            if len(self.data) < self.max_size:
                # Fill buffer
                self.data.append(image)
                result.append(image)
            else:
                if random.random() > 0.5:
                    # Use a stored image instead
                    idx = random.randint(0, self.max_size - 1)
                    old_image = self.data[idx].clone()
                    self.data[idx] = image  # Replace with new one
                    result.append(old_image)
                else:
                    # Use new image
                    result.append(image)
        return torch.cat(result, dim=0)  # Return batch

def plot_losses(loss_log, title="Training Losses", xlabel="Training Steps", ylabel="Loss", save_path=None):
    """
    Plots any number of named loss curves over time.

    Parameters:
    - loss_log: Dictionary with keys as loss names and values as lists of loss values.
    - title: Title of the plot.
    - xlabel: Label for x-axis.
    - ylabel: Label for y-axis.
    - save_path: If specified, saves the plot to this path.
    """
    plt.figure(figsize=(12, 6))

    for key, values in loss_log.items():
        plt.plot(values, label=key)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

import torch
from torchvision.utils import save_image
import os

def save_sample_images(tensor_images, epoch, output_dir="samples", num_images=64, filename_prefix="sample"):
    """
    Saves a grid of generated sample images.

    Args:
        tensor_images (torch.Tensor): A batch of images, shape (B, C, H, W), range [-1, 1] or [0, 1].
        epoch (int): The current epoch number (used for filename).
        output_dir (str): Directory where to save images.
        num_images (int): Number of images from the batch to save.
        filename_prefix (str): Prefix for the filename.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Select only first `num_images`
    tensor_images = tensor_images[:num_images]

    # Rescale from [-1, 1] to [0, 1] if needed
    if tensor_images.min() < 0:
        tensor_images = (tensor_images + 1) / 2

    save_path = os.path.join(output_dir, f"{filename_prefix}_epoch{epoch}.png")
    save_image(tensor_images, save_path, nrow=8, normalize=False)
    print(f"✅ Saved sample images to: {save_path}")


