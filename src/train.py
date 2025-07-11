import torch
import torch.nn
import torch.optim as optim
from tqdm import tqdm 
import os
from torchvision.utils import save_image

from generator import Generator
from discriminator import Discriminator
from dataset import MonetDataSet, MonetDataLoader
from torch.utils.data import transforms


# ===========================================
# CONFIG
# ===========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMAGE_SIZE = 128
NOISE_DIM = 100
EPOCHS = 100
LEARNING_RATE = 2e-4
FEATURES_GEN = 64
FEATURES_DISC = 64
SAVE_DIR = "outputs/samples"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================
# Transforms and Dataloader
# =========================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(0),
    transforms.Normalize([0.5], [0.5])
])

dataLoader = MonetDataLoader(folder_path="data/monet_jpg", transform=transform, batch_size=BATCH_SIZE)

# ==========================================
# Models
# ==========================================
G = Generator(noise_dim=NOISE_DIM, feature_maps=FEATURES_GEN).to(DEVICE)
D = Discriminator(feature_maps=FEATURES_DISC).to(DEVICE)

# ==========================================
# Loss & Optimizers
# ==========================================
criterion = torch.nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr = LEARNING_RATE, betas = (0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr = LEARNING_RATE, betas = (0.5, 0.999))

# Fixed noise for saving sample images
fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(DEVICE)

# ==========================================
# Training loop
# ==========================================

for epoch in range(EPOCHS):
    loop = tqdm(dataLoader, leave=True)
    for i, real in enumerate(loop): 
        real = real.to(DEVICE)
        batch_size = real.size(0)
        noise = torch.randn(batch_size, NOISE_DIM, 1, 1).to(DEVICE)

        # ================== Train Discriminator ==============
        fake = G(noise)
        D_real = D(real)
        D_fake = D(fake.detach())

        real_labels = torch.ones_like(D_real)
        fake_labels = torch.zeros_like(D_fake)

        loss_D_real = criterion(D_real, real_labels)
        loss_D_fake = criterion(D_fake, fake_labels)
        loss_D = loss_D_real + loss_D_fake

        D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        #================== Train Generator ==================
        output = D(fake)
        loss_G = criterion(output, real_labels)
        G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        #=======logging=========
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        loop.set_postfix(D_loss=loss_D.item(), G_loss=loss_G.item())
    
    with torch.no_grad():
        fake = G(fixed_noise)
        fake = fake*0.5+ 0.5
        save_image(fake, os.path.join(SAVE_DIR, f"epoch_{epoch+1}.png"), nrow=8)


