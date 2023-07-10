## Very simple implementation of a GAN, with architecture inspired by Goodfellow et al.
## GAN here used to generate BW images as in the MNIST database.
## Architecture:
## Generator: FCL -> LReLU -> FCL -> Tanh
## Discriminator: FCL -> LReLU -> FCL -> Sigmoid




import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import tqdm

class Generator(nn.Module):
    def __init__(self, NOISE_CHANNEL, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(NOISE_CHANNEL, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LR = 3e-4
NOISE_CHANNEL = 64
IMAGE_DIM = 28 * 28 * 1  # 784
BATCH_SIZE = 32
NUM_EPOCHS = 50

disc = Discriminator(IMAGE_DIM).to(device)
gen = Generator(NOISE_CHANNEL, IMAGE_DIM).to(device)
fixed_noise = torch.randn((BATCH_SIZE, NOISE_CHANNEL)).to(device)
transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# I've elected to save the dataset to a folder in the parent directory, so it can be accessed by all trailed GANS
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
dataset = datasets.MNIST(root=os.path.join(parent_dir, "dataset/"), transform=transforms, download=True)
loader = DataLoader(dataset, BATCH_SIZE=BATCH_SIZE, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), LR=LR)
opt_gen = optim.Adam(gen.parameters(), LR=LR)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

for epoch in range(NUM_EPOCHS):
    iterable = range(int(torch.ceil(torch.div(len(dataset), BATCH_SIZE)).item()))
    for batch_idx, (real, _) in enumerate(loader):
        for i in tqdm(iterable):
            real = real.view(-1, 784).to(device)
            BATCH_SIZE = real.shape[0]

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            noise = torch.randn(BATCH_SIZE, NOISE_CHANNEL).to(device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            # where the second option of maximizing doesn't suffer from
            # saturating gradients but they are mathematically similar
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1