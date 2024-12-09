# USE THIS
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import optuna
import optuna.visualization.matplotlib as vis_matplotlib

seed = 0
torch.manual_seed(0)

print("In VAE, running!")

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

# have prints output to both file and console (for Turing)
class DualOut: 
    def __init__(self, filepath): 
        self.terminal=sys.stdout
        self.logfile=open(filepath, "w")

    def write(self, message): 
        self.terminal.write(message) # console 
        self.logfile.write(message) # file 

    def flush(self): 
        self.terminal.flush()
        self.logfile.flush()

sys.stdout = DualOut(os.path.join(run_dir,"output.txt"))

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
        img_size=img_size

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

    def decode(self, points, img_size):
        points = self.decoder_input(points).view(-1, 128, img_size // 8, img_size // 8) 
        return self.decoder(points)

    def forward(self, x, img_size):
        mu, logvar = self.encode(x)
        points = self.reparameterize(mu, logvar)
        return self.decode(points, img_size), mu, logvar

# -- DISCRIMINATOR --
class Discriminator(nn.Module): 
    def __init__(self, img_size): 
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(256 * (img_size // 8) * (img_size // 8), 1),
            nn.Sigmoid()
        )
    
    def forward(self, x): 
        return self.model(x)

# -- LOSS FUNCTIONS --
def vae_loss(recon_x, x, mu, logvar, beta=0.1): # beta allows us to control ratio between recon loss and kl divergence
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_div

def gan_loss(d_real, d_fake): 
    real_loss = nn.BCELoss()(d_real, torch.ones_like(d_real))
    fake_loss = nn.BCELoss()(d_fake, torch.zeros_like(d_fake))
    return real_loss + fake_loss

def generator_loss(d_fake): 
    return nn.BCELoss()(d_fake, torch.ones_like(d_fake))

# -- TRAIN -- 
def train_vae_gan(img_size=128, batch_size=16, epochs=50, latent_dim=128, limit=None, learning_rate=1e-3, beta = 0.1, disc_ct=1):
    print(f"Params: \nImg size: {img_size}\nBatch size: {batch_size}\nEpochs: {epochs}\nLatent dim: {latent_dim}\nLimit: {limit}\nLearning Rate: {learning_rate}")
   
    # transform for preprocessing
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    dataset = MRIDataset('Data', transform=transform, limit=limit)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # initialize 
    vae = VAE(latent_dim=latent_dim, img_size=img_size).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=1e-5)

    discriminator = Discriminator(img_size).to(device)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=1e-5)

    # -- TRAIN LOOP --
    train_losses, val_losses, disc_losses = [], [], []
    for epoch in range(epochs):
        vae.train()
        discriminator.train()
        train_loss, disc_loss = 0, 0

        for batch in train_loader:
            batch = batch.to(device)

            # train vae
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch, img_size=img_size)
            vae_loss_val = vae_loss(recon_batch, batch, mu, logvar, beta=beta)
            vae_loss_val.backward()
            optimizer.step()
            train_loss += vae_loss_val.item()

            # train discriminator 
            for _ in range(disc_ct): 
                disc_optimizer.zero_grad()

                # generate fakes
                with torch.no_grad(): 
                    fake_points = torch.randn(batch.size(0), latent_dim).to(device)
                    fake_images = vae.decode(fake_points, img_size)

                # real and fake labels 
                real_labels = torch.ones(batch.size(0), 1).to(device)
                fake_labels = torch.zeros(batch.size(0), 1).to(device)

                # discriminator loss
                real_loss = F.binary_cross_entropy(discriminator(batch), real_labels)
                fake_loss = F.binary_cross_entropy(discriminator(fake_images), fake_labels)
                d_loss = real_loss + fake_loss  

                d_loss.backward()
                disc_optimizer.step()
                disc_loss += d_loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        disc_losses.append(disc_loss / (len(train_loader) * disc_ct))
        
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon_batch, mu, logvar = vae(batch, img_size=img_size)
                val_loss += vae_loss(recon_batch, batch, mu, logvar, beta=beta).item()

        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Discriminotor Loss: {disc_losses[-1]:.4f}")

    # plot training curves
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
    plt.savefig(os.path.join(run_dir, f"gen_img_{timestamp}.png"))
    plt.show()
    plt.clf()

    print(f"mu mean: {mu.mean()}, logvar mean: {logvar.mean()}")

    print(f"Val loss: {val_losses[-1]}")

    return train_losses[-1], val_losses[-1], disc_losses[-1]

# optimize hyperparameters with optuna 

def optimize_hyperparams(img_size=128, epochs=25, limit=100, n_trials=10):
    def objective(trial): 
        latent_dim = trial.suggest_int("latent_dim", 32, img_size) 
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3)
        beta = trial.suggest_float("beta", 0.1, 4.0)
        disc_ct = trial.suggest_int("disc_ct", 1, 10)

        vae = VAE(latent_dim=latent_dim, img_size=img_size).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

        train_loss, val_loss, disc_loss = train_vae_gan(
            img_size=img_size, 
            epochs=epochs, 
            latent_dim=latent_dim, 
            learning_rate=learning_rate, 
            beta=beta, limit=limit, 
            disc_ct=disc_ct)
        print(f"Completed trial!")
        return val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

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

    return study.best_params, study.best_value

best_params, best_value = optimize_hyperparams(
    img_size=128,
    epochs=5,
    limit=100,
    n_trials=2
)

print("Completed VAE")
print(f"Best parameters: {best_params}\n Best value: {best_value}\n")

# Usage
#root_dir = "Data"  # Replace with the actual path
#train_vae(root_dir, img_size=img_size, batch_size=16, epochs=10, latent_dim=128, learning_rate=1e-3, limit=100)

# restore prints 
sys.stdout = sys.__stdout__
