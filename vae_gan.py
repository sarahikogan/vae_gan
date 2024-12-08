import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# Define VAE-GAN Model
class VAE_GAN(nn.Module):
    def __init__(self, latent_dim, img_size):
        super(VAE_GAN, self).__init__()

        # VAE Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(128 * (img_size // 8) * (img_size // 8), latent_dim)
        self.fc_logvar = nn.Linear(128 * (img_size // 8) * (img_size // 8), latent_dim)

        # VAE Decoder (Generator)
        self.decoder_input = nn.Linear(latent_dim, 128 * (img_size // 8) * (img_size // 8))
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, img_size // 8, img_size // 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )

        # GAN Discriminator
        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * (img_size // 8) * (img_size // 8), 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = self.decoder_input(z)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# VAE Loss
def vae_loss(recon_x, x, mu, logvar, beta=1):
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

# GAN Loss (Discriminator)
def gan_loss(d_real, d_fake):
    real_loss = nn.BCELoss()(d_real, torch.ones_like(d_real))
    fake_loss = nn.BCELoss()(d_fake, torch.zeros_like(d_fake))
    return real_loss + fake_loss

# GAN Loss (Generator)
def generator_loss(d_fake):
    return nn.BCELoss()(d_fake, torch.ones_like(d_fake))

# Dataset Class for MRI Images
class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        self.root_dir = root_dir
        self.image_paths = []
        for subdir in ['Mild Dementia']:  # You can change this as needed
            subdir_path = os.path.join(root_dir, subdir)
            self.image_paths += [os.path.join(subdir_path, f) for f in os.listdir(subdir_path)]
        if limit:
            self.image_paths = self.image_paths[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        if self.transform:
            img = self.transform(img)
        return img

# Training Function
def train_vae_gan(model, train_loader, optimizer, vae_optimizer, device, epochs=50):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)

            # Zero the optimizers
            vae_optimizer.zero_grad()
            optimizer.zero_grad()

            # VAE Forward Pass
            recon_batch, mu, logvar = model(batch)
            vae_loss_value = vae_loss(recon_batch, batch, mu, logvar)

            # GAN Forward Pass
            d_real = model.discriminator(batch)
            d_fake = model.discriminator(recon_batch.detach())  # Discriminator on fake images
            g_fake = model.discriminator(recon_batch)  # Discriminator on fake generated images

            # Compute the discriminator and generator losses
            d_loss = gan_loss(d_real, d_fake)
            g_loss = generator_loss(g_fake)

            # Backpropagate VAE and GAN losses
            vae_loss_value.backward(retain_graph=True)
            d_loss.backward(retain_graph=True)
            g_loss.backward()

            # Update the parameters
            vae_optimizer.step()
            optimizer.step()

        print(f"Epoch [{epoch}/{epochs}], VAE Loss: {vae_loss_value.item()}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# Data Preprocessing and Loading
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Adjust normalization as needed
])

dataset = MRIDataset(root_dir='Data', transform=transform, limit=100)  # Use a subset for faster training
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, Optimizers, and Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 128  # Latent dimension size
model = VAE_GAN(latent_dim=latent_dim, img_size=128)  # Adjust image size as needed

vae_optimizer = optim.Adam(model.parameters(), lr=1e-2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_vae_gan(model, train_loader, optimizer, vae_optimizer, device, epochs=15)

# Generate new images
def generate_images(model, num_samples=4):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        generated_images = model.decode(z)
        
        # Save or plot generated images
        for i, img in enumerate(generated_images):
            plt.imshow(img.cpu().squeeze(), cmap='gray')
            plt.title(f"Generated image {i}")
            plt.savefig(f"generated_img_{i}.png")
            plt.show()

# Example: Generate some images after training
generate_images(model)
