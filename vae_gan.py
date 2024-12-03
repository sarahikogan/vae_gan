print("Starting imports")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from torchvision.io import read_image
from PIL import Image
from datetime import datetime
from torchvision.transforms import Resize
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import os

base_dir = "vae_runs"

# timestamped folders for run results
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join(base_dir, timestamp)
os.makedirs(run_dir, exist_ok=True)

print("Finished imports")

# load dataset
class MRIDataset(Dataset): 
    def __init__(self, root_dir, transform=None, limit=None): 
        self.root_dir = root_dir
        self.image_paths = []
        for subdir in ['Mild Dementia']: # 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']: # can choose just one of these to make it train quicker / on less data, i've been testing on only mild dementia
            subdir_path = os.path.join(root_dir, subdir)
            self.image_paths += [os.path.join(subdir_path, f) for f in os.listdir(subdir_path)]
        if limit: 
            self.image_paths = self.image_paths[:limit]  # limit images to train faster
        self.transform = transform 

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img

# vae class
class VAE(nn.Module):
    def __init__(self, input_channels, latent_dim, img_size):
        super(VAE, self).__init__()
        self.img_size = img_size

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )

        # flattened size after encoding
        dummy_input = torch.zeros(1, input_channels, img_size, img_size)
        conv_output_size = self.encoder(dummy_input).shape[1]

        self.fc_mu = nn.Linear(conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(conv_output_size, latent_dim)

        # decoder
        self.decoder_input = nn.Linear(latent_dim, conv_output_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, img_size // 8, img_size // 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, input_channels, 3, 1, 1), nn.Tanh()
        )


    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = self.fc_mu(encoded), self.fc_logvar(encoded)
        z = self.reparametrize(mu, logvar)
        decoded = self.decoder(self.decoder_input(z))
        return decoded, mu, logvar

# discriminator 
class Discriminator(nn.Module):
    def __init__(self, input_channels, img_size): 
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * (img_size // 8) * (img_size // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, x): 
        return self.model(x)

# calculate vae loss
def vae_loss(recon_x, x, mu, logvar, beta=1):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div / x.size(0)


# train vae
def train_vae(root_dir, img_size=128, batch_size=16, epochs=50, latent_dim=256, limit=None):

    epochs_str = ""
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # set data and convert to loaders
    dataset = MRIDataset(root_dir=root_dir, transform=transform, limit=limit)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    vae = VAE(input_channels=1, latent_dim=latent_dim, img_size=img_size).to(device)
    discriminator = Discriminator(input_channels=1, img_size = img_size).to(device)

    vae_optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)
    bce_loss = nn.BCELoss()

    # train loop
    train_losses, val_losses, d_losses = [], [], []
    for epoch in range(epochs):
        print("running epoch %d" % epoch)
        vae.train()
        discriminator.train()

        train_loss, d_loss_total, batch_ctr = 0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)

            # train vae 
            vae_optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch)
            vae_loss_val = vae_loss(recon_batch, batch, mu, logvar)

            # train discriminator 
            d_optimizer.zero_grad()
            real_labels = torch.ones(batch.size(0), 1).to(device)
            fake_labels = torch.zeros(batch.size(0), 1).to(device)

            # discriminator loss on real images 
            real_preds  = discriminator(batch)
            d_loss_real = bce_loss(real_preds, real_labels)

            # discriminator los on fake images 
            fake_preds = discriminator(batch)
            d_loss_fake = bce_loss(fake_preds, fake_labels)

            # total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            d_loss_total += d_loss.item()

            # train vae to fool discriminator! -- adversarial loss 
            adversarial_loss = bce_loss(discriminator(recon_batch))
            generator_loss = vae_loss_val + adversarial_loss 
            generator_loss.backward()
            vae_optimizer.step()

            train_loss += generator_loss.item()

        train_losses.append(train_loss / len(train_loader))
        d_losses.append(d_loss_total / len(train_loader))

        # validation loss 
        vae.eval()
        val_loss = 0
        with torch.no_grad(): 
            for batch in val_loader: 
                batch = batch.to(device)
                batch = batch.to(device)
                recon_batch, mu, logvar = vae(batch)
                val_loss += vae_loss(recon_batch, batch, mu, logvar).item()
        val_losses.append(val_loss / len(val_loader))

        # print progress 
        print(f"Train loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Discriminator Loss: {d_losses[-1]:.4f}")

    # training curves
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.legend()
    curve_plot_path = os.path.join(run_dir, "curve_plot.png")
    plt.savefig(curve_plot_path)
    plt.show()
    plt.close()
    
    vae.eval()
    sample_batch = next(iter(val_loader)).to(device)
    with torch.no_grad():
        recon_batch, _, _ = vae(sample_batch)

    # visualize reconstructions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    target_transform = Resize((img_size, img_size))

    for i in range(5):
        # show original image
        original_image = target_transform(to_pil_image(sample_batch[i].cpu().squeeze()))
        axes[0, i].imshow(original_image, cmap="gray")
        axes[0, i].set_title("Original")
        #axes[0, i].axis("off")

        # show reconstructed image
        axes[1, i].imshow(recon_batch[i].cpu().squeeze(), cmap="gray")
        axes[1, i].set_title("Reconstructed")
        #axes[1, i].axis("off")

    loss_plot_path = os.path.join(run_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.show()
    plt.clf()
    plt.close()

    model_save_path = os.path.join(run_dir, "vae_model.pth")
    torch.save(vae.state_dict(), model_save_path)

    metadata_path = os.path.join(run_dir, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write(f"Run Timestamp: {timestamp}\n")
        f.write(f"Latent Dimensions: {latent_dim}\n")
        f.write(f"Learning Rate: {vae_optimizer.defaults['lr']}\n")
        f.write(f"Epochs: {len(train_losses)}\n")
        f.write(f"Model Output: {epochs_str}")
        f.write(f"Trained on {limit} images")

# Run training
print("Training")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_vae(root_dir="Data", img_size=256, epochs=100, latent_dim=256, limit=500)
print("Finished training")

