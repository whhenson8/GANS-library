## More complex implementation of DCGAN, with architecture inspired by Radford 2015
## DCGAN here used to generate BW images as in the MNIST database.
## much more robust due to implementation of the various batch norms, and activation functions.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model_gp import Discriminator, Generator, initialize_weights
import os
from utils import gradient_penalty

# First defining hyperparameters with which to run the training.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4    # Suggested by some sources that this is a good place to start (extremely sensitive to HPs).
BATCH_SIZE = 64
IMAGE_SIZE = 64         # Change for other examples.
IMAGE_CHANNELS = 1      # Binary for the MNIST database 
NOISE_DIM = 100         # That suggested in the original paper (Named the Z dimension in the paper)
NUM_EPOCHS = 5          # Quite hefty so I have kept this small
FEATURES_DISC = 64      # Following that outlined in the paper. Number of features in the discriminator.
FEATURES_CRIT = 64      # Following that outlined in the paper. Number of features in the generator.
CRITIC_ITERATIONS = 5   ####################
LAMBDA_GP = 10

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(IMAGE_CHANNELS)], [0.5 for _ in range(IMAGE_CHANNELS)]
        ),
    ]
)

# If you train on MNIST, remember to set channels_img to 1
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
dataset = datasets.MNIST(root=os.path.join(parent_dir, "dataset/"), transform=transforms, download=True)
# In this example we use the MNIST database. Can be used to generate novel augmented data in medical images for eg.

# Pytorch implementation of the dataloader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(NOISE_DIM, IMAGE_CHANNELS, FEATURES_CRIT).to(device)
critic = Discriminator(IMAGE_CHANNELS, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0,0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0,0.9))

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    iterable = range(int(torch.ceil(torch.div(len(dataset), BATCH_SIZE)).item()))

    for batch_idx, (real, _) in enumerate(dataloader):
        for i in tqdm(iterable):
            real = real.to(device)
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
                fake = gen(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=device)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP*gp
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            output = critic(fake).reshape(-1)
            loss_gen = -torch.mean(output)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    fake = gen(fixed_noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1