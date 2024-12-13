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
import pydicom

base_dir = "vae_runs"

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join(base_dir, timestamp)
os.makedirs(run_dir, exist_ok=True)

print("Finished imports")

# loading dataset for T1
class MRIDataset(Dataset): 
    def __init__(self, root_dir, transform=None, limit=None): 
        self.root_dir = root_dir
        self.image_paths = []
        for subdir in ['Mild Dementia']: # 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']: # can choose just one of these to make it train quicker / on less data, i've been testing on only mild dementia
            subdir_path = os.path.join(root_dir, subdir)
            self.image_paths += [os.path.join(subdir_path, f) for f in os.listdir(subdir_path)]
        if limit: 
            self.image_paths = self.image_paths[:limit] 
        self.transform = transform 

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L") 
        if self.transform:
            img = self.transform(img)
        return img

# loading dataset for T2
class T2Dataset(Dataset): 
    def __init__(self, root_dir, transform=None, limit=None): 
        self.root_dir = root_dir
        self.image_paths = []
        for subdir in ['T2']: # specifically for T2 files
            subdir_path = os.path.join(root_dir, subdir)
            for file_name in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file_name)
                try:
                    pydicom.dcmread(file_path, stop_before_pixels=True) 
                    self.image_paths.append(file_path)
                except Exception:
                    continue
        if limit: 
            self.image_paths = self.image_paths[:limit] 
        self.transform = transform 

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            dicom = pydicom.dcmread(img_path)
            img = Image.fromarray(dicom.pixel_array).convert("L") 
        except Exception as e:
            raise ValueError(f"Error reading DICOM file {img_path}: {e}")
        
        if self.transform:
            img = self.transform(img)
        return img

# vae class
class MultiModalVAE(nn.Module):
    def __init__(self, input_channels, latent_dim, img_size):
        super(MultiModalVAE, self).__init__()
        self.img_size = img_size

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )

        # flattening size after encoding
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

# calculating vae loss
def vae_loss(recon_x, x, mu, logvar, beta=0.01):
    recon_loss = 0.7 * nn.MSELoss()(recon_x, x) + 0.3 * nn.L1Loss()(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div / x.size(0)

# Multi-modal training
class MultiModalDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return min(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        items = [dataset[idx] for dataset in self.datasets]
        combined = torch.cat(items, dim=0)  
        return combined

def train_multimodal_vae(root_dir1, root_dir2, img_size=128, batch_size=16, epochs=50, latent_dim=128, limit=None):
    print("Preparing datasets...")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset1 = MRIDataset(root_dir1, transform=transform, limit=limit)
    dataset2 = T2Dataset(root_dir2, transform=transform, limit=limit)
    multimodal_dataset = MultiModalDataset([dataset1, dataset2])

    print("Splitting datasets into training and validation...")
    train_size = int(0.8 * len(multimodal_dataset))
    val_size = len(multimodal_dataset) - train_size
    train_dataset, val_dataset = random_split(multimodal_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Initializing model...")
    vae = MultiModalVAE(input_channels=2, latent_dim=latent_dim, img_size=img_size).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=3e-4, weight_decay=1e-5)

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}")
        vae.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            print(f"  Processing batch {batch_idx + 1}/{len(train_loader)}")
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)
            print(f"    Batch loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        print(f"Finished Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}")

        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                print(f"  Validating batch {batch_idx + 1}/{len(val_loader)}")
                batch = batch.to(device)
                recon_batch, mu, logvar = vae(batch)
                loss = vae_loss(recon_batch, batch, mu, logvar)
                print(f"    Validation batch loss: {loss.item():.4f}")
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))
        print(f"Validation Loss: {val_losses[-1]:.4f}\n")

        # Generate and print a random image from the latent space after each epoch
        print("Generating random image from latent space...")
        random_latent_vector = torch.randn(1, latent_dim).to(device)
        with torch.no_grad():
            generated_image = vae.decoder(vae.decoder_input(random_latent_vector))
            generated_image = generated_image.squeeze().cpu()
            if len(generated_image.shape) == 3:
                generated_image = generated_image.mean(dim=0) 
            plt.imshow(generated_image, cmap="gray")
            plt.title(f"Random Image - Epoch {epoch + 1}")
            plt.axis("off")
            plt.show()

        # Plotting training and validation loss curves
    print("Plotting training and validation loss curves...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    print("Training Complete!")

# Multi-modal training
print("Training multi-modal VAE")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_multimodal_vae(
    root_dir1="/Users/dominicranelli/Downloads/Data/",
    root_dir2="/Users/dominicranelli/Downloads/Neurohacking_data/BRAINIX/DICOM/",
    img_size=128,
    batch_size=16,
    epochs=3000,
    latent_dim=128,
    limit=1000
)
print("Finished training multi-modal VAE")