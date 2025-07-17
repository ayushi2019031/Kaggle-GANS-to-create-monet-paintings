'''
This file is the code submitted to kaggle to make inference and generate the images.zip file. 
Adding it here for future reference. 

Baseline score [DCGANs] = 312.89
'''

device = "cuda" if torch.cuda.is_available() else "cpu"

from tqdm import tqdm
import os
from PIL import Image
import zipfile
import io

import torch
import torch.nn as nn

gen = Generator()
gen.load_state_dict(torch.load("/kaggle/input/dcgan-checkpoint/pytorch/99-checkpoint/1/gan_checkpoint_99.pth", map_location='cpu')['generator'])
print(gen.eval())


batch_size = 100
z_dim = 100  # latent dim
num_images = 10_000

zip_path = "images.zip"
zipf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)

idx = 0
with torch.no_grad():
    for i in tqdm(range(0, num_images, batch_size)):
        z = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_imgs = gen(z)
        fake_imgs = (fake_imgs + 1) / 2  # scale from [-1, 1] to [0, 1]

        fake_imgs = (fake_imgs * 255).clamp(0, 255).byte().cpu()  # shape: [B, 3, H, W]
        for img_tensor in fake_imgs:
            idx += 1
            img_pil = Image.fromarray(img_tensor.permute(1, 2, 0).numpy())  # CHW → HWC
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format='PNG')
            zipf.writestr(f'image_{idx:05d}.png', img_bytes.getvalue())
            idx += 1

zipf.close()

!zip -rq /kaggle/working/images.zip /kaggle/working/images