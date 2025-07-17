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
