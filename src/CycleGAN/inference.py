import os
import torch
from torchvision import transforms
from PIL import Image
from generator import Generator  # Or replace with your actual class

# ===== CONFIG =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/G_AB_epoch_1.pth"
INPUT_DIR = "../../data/gan-getting-started/photo_jpg"
OUTPUT_DIR = "generated_images"
IMAGE_SIZE = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# ===== LOAD MODEL =====
model = Generator()
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

# ===== INFERENCE =====
with torch.no_grad():
    for file in os.listdir(INPUT_DIR):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(INPUT_DIR, file)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        output_tensor = model(input_tensor).squeeze(0).cpu()
        output_tensor = (output_tensor + 1) / 2  # Normalize from [-1,1] to [0,1]

        output_image = transforms.ToPILImage()(output_tensor)
        output_path = os.path.join(OUTPUT_DIR, f"monet_{file}")
        output_image.save(output_path)

        print(f"Saved: {output_path}")
