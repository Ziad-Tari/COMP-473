from ..models.discriminant import Discriminator
from ..models.gen import Generator
from ..preprocessing.data_loader import get_dataloader

import torch

netG = Generator(ngpu=1)
netD = Discriminator(ngpu=1)

dataloader, dataset = get_dataloader('celeba')

# sample input 100-dimensional uniform distribution vector
fixed_noise = torch.randn(64, 100, 1, 1)

for epoch in range(5):
    for batch_idx, (real_images, _) in enumerate(dataloader):
        
        # real_images.shape = (128, 3, 64, 64)        
        batch_size = real_images.size(0)  # 128

        # ...        