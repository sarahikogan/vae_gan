# USE THIS
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import optuna

seed = 0
torch.manual_seed(0)

print("In VAE, running! ")

# timestamped folders for run results
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join("vae_runs", timestamp)
os.makedirs(run_dir, exist_ok=True)

# -- SET DEVICE --
if torch.cuda.is_available(): 
    device = torch.device("cuda")
    print("Using GPU!!!")
else: 
    device = "cpu"

img_size = 400
sys.stdout = open(os.path.join(run_dir, "output.txt"), "w") # redirect prints to file if running in Turing

# -- DEFINE MRI DATA --
class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        self.root_dir = root_dir
        self.image_paths = []
        for subdir in ['Mild Dementia']: # TODO switch to all patients - learn representations from all stages of dementia
            subdir_path = os.path.join(root_dir, subdir)
            self.image_paths += [os.path.join(subdir_path, f) for f in os.listdir(subdir_path)]
        if limit:
            self.image_paths = self.image_paths[:limit]  # if neccesary, set limit on training to run model faster
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img

# -- VAE MODEL --
class VAE(nn.Module):
    def __init__(self, latent_dim, img_size):
        super(VAE, self).__init__()

        # encode
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * (img_size // 8) * (img_size // 8), latent_dim)
        self.fc_logvar = nn.Linear(128 * (img_size // 8) * (img_size // 8), latent_dim)
        
        # decode
        self.decoder_input = nn.Linear(latent_dim, 128 * (img_size // 8) * (img_size // 8))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Tanh()
        )
        
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, img_size):
        z = self.decoder_input(z).view(-1, 128, img_size // 8, img_size // 8) # 
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, img_size), mu, logvar

# loss function
def vae_loss(recon_x, x, mu, logvar, beta=0.1): # beta allows us to control ratio between recon loss and kl divergence
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_div

# train 
def train_vae(root_dir, img_size=128, batch_size=16, epochs=50, latent_dim=128, limit=None, learning_rate=1e-3, beta = 0.1):
    print(f"Params: \nImg size: {img_size}\nBatch size: {batch_size}\nEpochs: {epochs}\nLatent dim: {latent_dim}\nLimit: {limit}\nLearning Rate: {learning_rate}")
   
    # transform for preprocessing
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    dataset = MRIDataset(root_dir=root_dir, transform=transform, limit=limit)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # initialize
    vae = VAE(latent_dim=latent_dim, img_size=img_size).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=1e-5)

    # -- TRAIN LOOP --
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        vae.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch)

            loss = vae_loss(recon_batch, batch, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon_batch, mu, logvar = vae(batch)

                loss = vae_loss(recon_batch, batch, mu, logvar)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # -- PLOT TRAINING CURVES --
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "training_curve.png"))
    plt.show()
    plt.clf()

    torch.save(vae.state_dict(), os.path.join(run_dir,"vae_model.pth"))

    # visualization from random latent points - check how good the latent space is 
    vae.eval()
    num_samples = 16
    with torch.no_grad():
        points = torch.randn(num_samples, latent_dim).to(device)
        generated_images = vae.decode(points, img_size).cpu()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i].squeeze(), cmap="gray")
        ax.axis("off")
    plt.savefig(os.path.join(run_dir, "gen_img.png"))
    plt.show()
    plt.clf()


    print(f"Val loss: {val_losses[-1]}")

    return train_losses[-1], val_losses[-1]

# optimize hyperparameters with optuna 
def objective(trial): 
    latent_dim = trial.suggest_int("latent_dim", 32, img_size) 
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
    beta = trial.suggest_float("beta", 0.1, 4.0)

    vae = VAE(latent_dim=latent_dim, img_size=img_size).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    train_loss, val_loss = train_vae("Data", img_size=img_size, epochs=50, latent_dim=latent_dim, learning_rate=learning_rate, beta=beta, limit=100)
    print(f"Completed trial!")
    return val_loss

pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(direction='minimize', pruner=pruner)
study.optimize(objective, n_trials=100)

print(f"Best hyperparameters: {study.best_params}")

optuna.visualization.matplotlib.plot_optimization_history(study)
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "optimization_history.png"))
plt.clf()

optuna.visualization.matplotlib.plot_param_importances(study)
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "param_importances.png"))
plt.clf()

optuna.visualization.matplotlib.plot_parallel_coordinate(study)
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "parallel_coords.png"))
plt.clf()

optuna.visualization.matplotlib.plot_timeline(study)
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "plot_timeline.png"))
plt.clf()


# Usage
#root_dir = "Data"  # Replace with the actual path
#train_vae(root_dir, img_size=img_size, batch_size=16, epochs=10, latent_dim=128, learning_rate=1e-3, limit=100)

# restore prints 
sys.stdout = sys.__stdout__
