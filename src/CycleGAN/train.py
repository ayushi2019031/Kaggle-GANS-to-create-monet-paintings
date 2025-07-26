import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from generator import Generator, UNetGenerator
from discriminator import Discriminator
from dataset import MonetDataLoader
from utils import ReplayBuffer, save_sample_images  # Optional
import itertools
from torchvision import transforms

import glob
import re

def load_latest_checkpoint(model, optimizer, name, checkpoint_dir="checkpoints/CYCLEGAN"):
    checkpoint_files = glob.glob(f"{checkpoint_dir}/{name}_epoch_*.pth")
    if not checkpoint_files:
        print(f"No checkpoint found for {name}. Starting from scratch.")
        return model, optimizer, 1  # start at epoch 1

    # Extract epoch numbers
    def extract_epoch(path):
        match = re.search(r"epoch_(\d+)", path)
        return int(match.group(1)) if match else -1

    latest_ckpt = max(checkpoint_files, key=extract_epoch)
    epoch = extract_epoch(latest_ckpt)

    checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    print(f"✔ Loaded {name} from epoch {epoch}")
    return model, optimizer, epoch + 1


# === Config ===
EPOCHS = 200
BATCH_SIZE = 1
LR = 0.0002
IMAGE_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Models ===
G_AB = UNetGenerator().to(DEVICE)  # Photo → Monet
G_BA = UNetGenerator().to(DEVICE)  # Monet → Photo
D_A = Discriminator().to(DEVICE)  # Real vs fake photo
D_B = Discriminator().to(DEVICE)  # Real vs fake Monet

# === Losses ===
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# === Optimizers ===
optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=LR, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=LR, betas=(0.5, 0.999))

G_AB, optimizer_G, start_epoch = load_latest_checkpoint(G_AB, optimizer_G, "G_AB")
G_BA, optimizer_G, _ = load_latest_checkpoint(G_BA, optimizer_G, "G_BA")
D_A, optimizer_D_A, _ = load_latest_checkpoint(D_A, optimizer_D_A, "D_A")
D_B, optimizer_D_B, _ = load_latest_checkpoint(D_B, optimizer_D_B, "D_B")

# =========================================
# Transforms and Dataloader
# =========================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # ensure all images are of same size as model expects a fixed size input.
    transforms.ToTensor(), # converts from PIL.Image to torch.tensor
    transforms.Normalize([0.5], [0.5]) # normalize the images values to be between [-1.0, + 1.0]
])

# === Data ===
dataloader = MonetDataLoader(folder_path="../../data/gan-getting-started", transform=transform, batch_size=BATCH_SIZE).get_data_loader()

# === Buffers for stable training ===
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# === Training Loop ===
for epoch in range(start_epoch, EPOCHS+1):
    
    for i, batch in enumerate(dataloader):
        real_A = batch['A'].to(DEVICE)  # Real photo
        real_B = batch['B'].to(DEVICE)  # Real Monet

        # === Train Generators ===
        optimizer_G.zero_grad()

        # Identity loss: G_AB(B) ≈ B, G_BA(A) ≈ A
        idt_A = G_BA(real_A)
        idt_B = G_AB(real_B)
        loss_identity = criterion_identity(idt_A, real_A) + criterion_identity(idt_B, real_B)

        # GAN loss: D_B(G_AB(A)) should think it's real Monet
        fake_B = G_AB(real_A)
        pred_fake_B = D_B(fake_B)
        loss_GAN_AB = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

        # GAN loss: D_A(G_BA(B)) should think it's real photo
        fake_A = G_BA(real_B)
        pred_fake_A = D_A(fake_A)
        loss_GAN_BA = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

        # Cycle-consistency loss: G_BA(G_AB(A)) ≈ A
        rec_A = G_BA(fake_B)
        rec_B = G_AB(fake_A)
        loss_cycle = criterion_cycle(rec_A, real_A) + criterion_cycle(rec_B, real_B)

        # Total generator loss
        lambda_cycle = 10.0
        lambda_idt = 5.0
        loss_G = loss_GAN_AB + loss_GAN_BA + lambda_cycle * loss_cycle + lambda_idt * loss_identity
        loss_G.backward()
        optimizer_G.step()

        # === Train Discriminator A (real photo vs fake photo) ===
        optimizer_D_A.zero_grad()
        loss_real_A = criterion_GAN(D_A(real_A), torch.ones_like(D_A(real_A)))
        fake_A_detached = fake_A_buffer.push_and_pop(fake_A)
        loss_fake_A = criterion_GAN(D_A(fake_A_detached.detach()), torch.zeros_like(D_A(fake_A_detached)))
        loss_D_A = 0.5 * (loss_real_A + loss_fake_A)
        loss_D_A.backward()
        optimizer_D_A.step()

        # === Train Discriminator B (real Monet vs fake Monet) ===
        optimizer_D_B.zero_grad()
        loss_real_B = criterion_GAN(D_B(real_B), torch.ones_like(D_B(real_B)))
        fake_B_detached = fake_B_buffer.push_and_pop(fake_B)
        loss_fake_B = criterion_GAN(D_B(fake_B_detached.detach()), torch.zeros_like(D_B(fake_B_detached)))
        loss_D_B = 0.5 * (loss_real_B + loss_fake_B)
        loss_D_B.backward()
        optimizer_D_B.step()

        # === Logging ===
        if i % 100 == 0:
            print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                  f"[D_A: {loss_D_A.item():.2f}] [D_B: {loss_D_B.item():.2f}] "
                  f"[G: {loss_G.item():.2f}] [Cycle: {loss_cycle.item():.2f}] [ID: {loss_identity.item():.2f}]")

    # === Save image samples ===
    if epoch % 5 == 0:
        save_sample_images(G_AB, G_BA, dataloader, epoch, DEVICE)

    # === Checkpoints ===
    if epoch % 25 == 0 or epoch <= 1:
        os.makedirs("checkpoints/CYCLEGAN", exist_ok=True)

        torch.save({"model_state": G_AB.state_dict(), "optimizer_state": optimizer_G.state_dict()},
                f"checkpoints/CYCLEGAN/G_AB_epoch_{epoch}.pth")
        torch.save({"model_state": G_BA.state_dict(), "optimizer_state": optimizer_G.state_dict()},
                f"checkpoints/CYCLEGAN/G_BA_epoch_{epoch}.pth")
        torch.save({"model_state": D_A.state_dict(), "optimizer_state": optimizer_D_A.state_dict()},
                f"checkpoints/CYCLEGAN/D_A_epoch_{epoch}.pth")
        torch.save({"model_state": D_B.state_dict(), "optimizer_state": optimizer_D_B.state_dict()},
                f"checkpoints/CYCLEGAN/D_B_epoch_{epoch}.pth")


        os.makedirs("checkpoints", exist_ok=True)
        torch.save(G_AB.state_dict(), f"checkpoints/G_AB_epoch_{epoch}.pth")
        torch.save(G_BA.state_dict(), f"checkpoints/G_BA_epoch_{epoch}.pth")
        torch.save(D_A.state_dict(), f"checkpoints/D_A_epoch_{epoch}.pth")
        torch.save(D_B.state_dict(), f"checkpoints/D_B_epoch_{epoch}.pth")

loss_log = {'G': [], 'D_A': [], 'D_B': [], 'cycle': [], 'identity': []}

# during training:
loss_log['G'].append(loss_G.item())
loss_log['D_A'].append(loss_D_A.item())
loss_log['D_B'].append(loss_D_B.item())
loss_log['cycle'].append(loss_cycle.item())
loss_log['identity'].append(loss_identity.item())
