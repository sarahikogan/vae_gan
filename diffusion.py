import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# Define a Simple Diffusion Model
class DiffusionModel(nn.Module):
    def __init__(self, img_size, timesteps=1000):
        super(DiffusionModel, self).__init__()
        self.img_size = img_size
        self.timesteps = timesteps

        # A simple UNet-like architecture for diffusion denoising
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x, t):
        # Add time-embedding awareness if needed
        return self.network(x)


# VAE-GAN Model


# Train the Diffusion Model
def train_diffusion_model(diffusion_model, train_loader, optimizer, device, timesteps=1000, epochs=10):
    print("Starting diffusion model training...")
    diffusion_model.to(device)
    mse_loss = nn.MSELoss()

    # Generate noise schedule
    beta = torch.linspace(0.0001, 0.02, timesteps).to(device)
    alpha = 1 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}...")
        diffusion_model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            batch = batch.to(device)

            # Sample random timesteps for each batch
            t = torch.randint(0, timesteps, (batch.size(0),), device=device).long()

            # Add noise to the images
            noise = torch.randn_like(batch).to(device)
            noisy_images = (
                torch.sqrt(alpha_hat[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)) * batch +
                torch.sqrt(1 - alpha_hat[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)) * noise
            )

            # Predict the noise added by the diffusion process
            predicted_noise = diffusion_model(noisy_images, t)

            # Compute loss
            loss = mse_loss(predicted_noise, noise)
            epoch_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} Loss: {epoch_loss / len(train_loader):.4f}")

    print("Diffusion model training completed.")


# Integrating VAE-GAN and Diffusion Models
def generate_with_diffusion(vae_gan, diffusion_model, latent_dim, device, num_samples=4, timesteps=1000):
    vae_gan.eval()
    diffusion_model.eval()

    print("Generating images with VAE-GAN and Diffusion Model...")
    with torch.no_grad():
        # Sample from the latent space
        z = torch.randn(num_samples, latent_dim).to(device)
        generated_images = vae_gan.decode(z)

        # Diffusion refinement process
        beta = torch.linspace(0.0001, 0.02, timesteps).to(device)
        alpha = 1 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)

        # Start with noisy images
        x_t = generated_images.clone()
        for t in reversed(range(timesteps)):
            noise = torch.randn_like(x_t) if t > 0 else 0
            x_t = (
                1 / torch.sqrt(alpha[t]) * (x_t - (1 - alpha[t]) / torch.sqrt(1 - alpha_hat[t]) * diffusion_model(x_t, t))
                + torch.sqrt(beta[t]) * noise
            )

        # Visualize generated images
        for i, img in enumerate(x_t):
            plt.imshow(img.cpu().squeeze(), cmap='gray')
            plt.title(f"Generated Image {i + 1}")
            plt.axis("off")
            plt.show()


# Example Usage
print("Setting up data pipeline...")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MRIDataset(root_dir="/Users/dominicranelli/Downloads/Data/", transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
print("Data pipeline set up.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128

print("Initializing models and optimizers...")
vae_gan = VAE_GAN(latent_dim=latent_dim, img_size=128)
vae_gan_optimizer = optim.Adam(vae_gan.parameters(), lr=1e-3)

diffusion_model = DiffusionModel(img_size=128, timesteps=1000)
diffusion_optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-4)

print("Models and optimizers initialized.")

# Train VAE-GAN
fid_scores, mse_scores = train_vae_gan(vae_gan, train_loader, vae_gan_optimizer, vae_gan_optimizer, device, latent_dim, epochs=30)

# Train Diffusion Model
train_diffusion_model(diffusion_model, train_loader, diffusion_optimizer, device, timesteps=1000, epochs=10)

# Generate Images
generate_with_diffusion(vae_gan, diffusion_model, latent_dim, device, num_samples=4, timesteps=1000)