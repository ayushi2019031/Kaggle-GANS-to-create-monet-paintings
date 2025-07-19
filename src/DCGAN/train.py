import torch
import torch.nn
import torch.optim as optim
from tqdm import tqdm 
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from generator import Generator
from discriminator import Discriminator
from DCGAN.dataset import MonetDataSet, MonetDataLoader
from torchvision import transforms

import yaml
import argparse

## Load the config for the path mentioned. 
def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Parse the argument to pass in custom configs as per environment. 
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/default.yaml', help='Path to config file')
args = parser.parse_args()

config = load_config(args.config)


# ===========================================
# CONFIG
# ===========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = config['training']['batch_size']
IMAGE_SIZE = config['data']['image_size']
NOISE_DIM = config['model']['z_dim']
EPOCHS = config['training']['num_epochs']
LEARNING_RATE = config['training']['learning_rate']
FEATURES_GEN = config['model']['feature_maps_gen']
FEATURES_DISC = config['model']['feature_maps_disc']
SAVE_DIR = config['save']['output_dir']
INPUT_FOLDER_PATH = config['data']['dataset_path']

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================
# Transforms and Dataloader
# =========================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # ensure all images are of same size as model expects a fixed size input.
    transforms.ToTensor(), # converts from PIL.Image to torch.tensor
    transforms.Normalize([0.5], [0.5]) # normalize the images values to be between [-1.0, + 1.0]
])

dataLoader = MonetDataLoader(folder_path="data/gan-getting-started/monet_jpg/", transform=transform, batch_size=BATCH_SIZE).get_data_loader()

# ==========================================
# Models
# ==========================================

G = Generator(noise_dim=NOISE_DIM, feature_maps=FEATURES_GEN).to(DEVICE)
D = Discriminator(feature_maps=FEATURES_DISC).to(DEVICE)

# ==========================================
# Loss & Optimizers
# ==========================================

name_of_loss_criterion = {"BCE": torch.nn.BCELoss, "WassLoss": -torch.mean}
criterion = name_of_loss_criterion["BCE"] # Binary cross entropy loss
optimizer_G = optim.Adam(G.parameters(), lr = LEARNING_RATE, betas = (0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr = LEARNING_RATE, betas = (0.5, 0.999))

# Fixed noise for saving sample images
fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(DEVICE)

# Make a note of the losses in generator and discriminator model while training 
# for understanding progression of GANs. 
G_losses = []
D_losses = []

# plot losses function
def plot_losses(G_losses, D_losses, save_path='outputs/loss_curve.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label="Generator Loss", linewidth=2)
    plt.plot(D_losses, label="Discriminator Loss", linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss During Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# Training loop
# ==========================================

for epoch in range(EPOCHS):

    checkpoint_path = f"outputs/gan_checkpoint_epoch_{epoch+1}.pth"
    if os.path.exists(checkpoint_path):
        print(f"[Epoch {epoch+1}] Checkpoint exists. Skipping training.")
        continue  # Skip this epoch, already done

    print(f"[Epoch {epoch+1}] Starting training...")

    loop = tqdm(dataLoader, leave=True) # tqdm shows progress bar for loops. 
    for i, real in enumerate(loop): 
        real = real.to(DEVICE)
        batch_size = real.size(0) # get batch size
        noise = torch.randn(batch_size, NOISE_DIM, 1, 1).to(DEVICE) # generate noise randomly.

        # ================== Train Discriminator ==============
        fake = G(noise) # generate fake images by the generator. 
        D_real = D(real) # get predictions for real images
        D_fake = D(fake.detach()) # get predictions for fake images

        real_labels = torch.ones_like(D_real) # all real images true probability value is 1.0
        fake_labels = torch.zeros_like(D_fake) # all fake images true probability value is 0.0

        loss_D_real = criterion(D_real, real_labels) # compute loss for real images of discriminator
        loss_D_fake = criterion(D_fake, fake_labels) # compute loss for fake images of discriminator
        loss_D = loss_D_real + loss_D_fake # compute total loss - real loss + fake loss

        D.zero_grad() # clear all previous gradients
        loss_D.backward() # compute the gradients
        optimizer_D.step() # update the model weights using the Adam optimizer

        #================== Train Generator ==================
        output = D(fake) # generate output probabilities from the discriminator using the fake images generated by GAN. 
        loss_G = criterion(output, real_labels) # compute BCE loss between output and fake labelsl
        G.zero_grad() # clear all the past gradients of the generator
        loss_G.backward() # compute the gradients
        optimizer_G.step() # update the model weights using the Adam optimizer.

        #=======logging=========
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        loop.set_postfix(D_loss=loss_D.item(), G_loss=loss_G.item())
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

    
    # save sample images at every epcoh
    with torch.no_grad():
        fake = G(fixed_noise) 
        fake = fake*0.5+ 0.5
        save_image(fake, os.path.join(SAVE_DIR, f"epoch_{epoch+1}.png"), nrow=8)

        torch.save({
        'generator': G.state_dict(),
        'discriminator': D.state_dict(),
        'g_optimizer': optimizer_G.state_dict(),
        'd_optimizer': optimizer_D.state_dict(),
    }, f'outputs/gan_checkpoint_{epoch+1}.pth')
        
plot_losses(G_losses=G_losses, D_losses=D_losses, save_path=os.path.join(SAVE_DIR, "loss_curve.png"))


from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

fid = FrechetInceptionDistance(feature=2048).to(DEVICE)
inception = InceptionScore().to(DEVICE)

FID_scores = []
IS_scores = []

# Evaluate FID and IS
with torch.no_grad():
    fake_images = G(fixed_noise)
    fake_images = fake_images * 0.5 + 0.5  # unnormalize to [0, 1]

    # Resize to 299x299 for InceptionV3 if necessary
    resized_fake = torch.nn.functional.interpolate(fake_images, size=(299, 299), mode='bilinear')

    fid.update(real, real=True)
    fid.update(resized_fake, real=False)

    inception.update(resized_fake)

    fid_score = fid.compute().item()
    is_score, _ = inception.compute()
    is_score = is_score.item()

    FID_scores.append(fid_score)
    IS_scores.append(is_score)

    print(f"[Epoch {epoch+1}] FID: {fid_score:.2f}, IS: {is_score:.2f}")

    # Reset states
    fid.reset()
    inception.reset()

def plot_fid_is(FID_scores, IS_scores, save_path='outputs/fid_is_curve.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(FID_scores, label="FID Score ↓", color='red', linewidth=2)
    plt.plot(IS_scores, label="Inception Score ↑", color='green', linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("FID and Inception Scores During Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

plot_fid_is(FID_scores, IS_scores, save_path=os.path.join(SAVE_DIR, "fid_is_curve.png"))
