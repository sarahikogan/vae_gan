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
import subprocess

seed = 0
torch.manual_seed(0)

print("In VAE, running!")

# timestamped folders for run results
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join("vae_gan3_runs", timestamp)
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
        self.terminal = sys.stdout  # Capture the terminal output
        self.logfile = open(filepath, "w")  # Open the output file to write
        self.errorfile = open(filepath.replace("output.txt", "error.txt"), "w")  # Capture error logs

    def write(self, message):
        self.terminal.write(message)  # Print to terminal
        self.logfile.write(message)  # Write to output file
        if message.startswith('Traceback'):  # If it's an error message, also log to error file
            self.errorfile.write(message)

    def flush(self):
        self.terminal.flush()  # Ensure terminal output is flushed
        self.logfile.flush()  # Ensure log file output is flushed
        self.errorfile.flush()  # Ensure error log is flushed

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
            SelfAttention(32),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            SelfAttention(64),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            SelfAttention(128),
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

    def forward(self, x, img_size):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, img_size), mu, logvar
    
# -- SELF-ATTENTION LAYERS -- 
class SelfAttention(nn.Module):

    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels//8, 1)
        self.key = nn.Conv2d(channels, channels//8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x): 
        batch_size, C, H, W = x.size()

        query = self.query(x).view(batch_size, -1, H*W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H*W)
        attention=torch.softmax(torch.bmm(query,key), dim=-1) 
        value = self.value(x).view(batch_size, -1, H*W) 

        out=torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, C, H, W)
        out = self.gamma * out + x
        return out

# -- GENERATOR FOR GAN --
class Generator(nn.Module):
    def __init__(self, latent_dim, img_size): 
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, 128*(img_size//8) * (img_size // 8))
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 4, 2, 1)
    
    def forward(self, z, img_size): 
        z = self.fc(z).view(-1, 128, img_size//8, img_size//8)
        x = torch.relu(self.deconv1(z))
        x = torch.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))
        return x
    
# -- DISCRIMINATOR --
class Discriminator(nn.Module): 
    def __init__(self, img_size): 
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.fc = nn.Linear(128 * (img_size // 8) * (img_size // 8), 1)
    
    def forward(self, x): 
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # flatten
        return torch.sigmoid(self.fc(x))

# -- LOSS FUNCTIONS --
def wgan_loss_d(discriminator, real, fake, real_images, fake_images, epsilon=1e-6): 

    # wasserstein loss - more stable training process
    loss_real = torch.mean(real)
    loss_fake = torch.mean(fake)

    # gradient penalty - helps to prevent discriminator from becoming too confident in its predictions
    epsilon = torch.rand(real_images.size(0), 1, 1, 1).to(device)
    interpolated = epsilon * real_images + (1-epsilon) * fake_images
    interpolated.requires_grad_(True)

    interpolated_pts = discriminator(interpolated)
    gradients = torch.autograd.grad(outputs=interpolated_pts, inputs=interpolated, grad_outputs=torch.ones_like(interpolated_pts).to(device),create_graph=True, retain_graph=True)[0]
    gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

    # loss w/ penalty: 
    d_loss = loss_fake - loss_real + 10 * gradient_penalty 
    return d_loss

def wgan_loss_g(fake): 
    return -torch.mean(fake)

def vae_loss(recon_x, x, mu, logvar, beta=0.1): # beta allows us to control ratio between recon loss and kl divergence
    recon_loss = nn.MSELoss()(recon_x, x)
    logvar = torch.clamp(logvar, min=-10, max=10)  # Clamp logvar
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_div

def gan_loss_d(real, fake): 
    real_loss = nn.BCELoss()(real, torch.ones_like(real)) # real images labeled 1
    fake_loss = nn.BCELoss()(fake, torch.zeros_like(fake)) # fake images labeled 0
    return (real_loss + fake_loss) / 2

def gan_loss_g(fake):
    return nn.BCELoss()(fake, torch.ones_like(fake)) # fake images w/ 1 label

# -- TRAIN -- 
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

    os.makedirs('generate_images', exist_ok=True)
    # -- TRAIN LOOP --
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        vae.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch, img_size=img_size)

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
                recon_batch, mu, logvar = vae(batch, img_size=img_size)

                loss = vae_loss(recon_batch, batch, mu, logvar)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

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

    return train_losses[-1], val_losses[-1]

def train_vae_gan(img_size=128, batch_size=16, epochs=50, latent_dim=128, limit=1000, learning_rate=1e-3, beta=0.1): 
    print(f"Params: \nImg size: {img_size}\nBatch size: {batch_size}\nEpochs: {epochs}\nLatent dim: {latent_dim}\nLimit: {limit}\nLearning Rate: {learning_rate}\nBeta: {beta}")
    
    # transform for preprocessing 
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = MRIDataset("Data", transform=transform, limit=limit)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # initialize models
    vae = VAE(latent_dim=latent_dim, img_size=img_size).to(device)
    generator = Generator(latent_dim=latent_dim, img_size=img_size).to(device)
    discriminator = Discriminator(img_size=img_size).to(device)

    optimizer_vae = optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_gen = optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=1e-5)

    # -- TRAIN LOOP --
    train_losses, val_losses = [], []
    for epoch in range(epochs): 
        vae.train()
        generator.train()
        discriminator.train()

        train_loss = 0
        for batch in train_loader: 
            batch = batch.to(device)

            # get real and fake images
            real_labels = discriminator(batch)
            noise = torch.randn(batch.size(0), latent_dim).to(device)
            fake_images = generator(noise, img_size)
            fake_labels = discriminator(fake_images.detach())

            # discriminator loss 
            disc_loss = wgan_loss_d(discriminator, real_labels, fake_labels, batch, fake_images)
            optimizer_disc.zero_grad()
            disc_loss.backward()
            optimizer_disc.step()

            # train generator
            optimizer_gen.zero_grad()
            fake_labels = discriminator(fake_images)

            # generator loss
            gen_loss = wgan_loss_g(fake_labels)
            gen_loss.backward()
            optimizer_gen.step()

            # train vae 
            optimizer_vae.zero_grad()
            recon_batch, mu, logvar = vae(batch, img_size=img_size)
            vae_loss_val = vae_loss(recon_batch, batch, mu, logvar, beta=beta)
            vae_loss_val.backward()
            optimizer_vae.step()

            train_loss += disc_loss.item() + gen_loss.item() + vae_loss_val.item()

        train_losses.append(train_loss/len(train_loader))

        # validation loop
        vae.eval()
        generator.eval()
        discriminator.eval()

        val_loss = 0
        with torch.no_grad(): 
            for batch in val_loader: 
                batch = batch.to(device)

                # vae loss for validation set 
                recon_batch, mu, logvar = vae(batch, img_size=img_size)
                val_loss_val = vae_loss(recon_batch, batch, mu, logvar, beta=beta)
                val_loss += val_loss_val.item()

        val_losses.append(val_loss/len(val_loader))
        print(f"Epoch {epoch}/{epochs}, train loss: {train_loss/len(train_loader)}, val loss: {val_loss/len(val_loader)}")

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.savefig("training_curve.png")
    plt.show()

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



    return val_losses[-1]

# optimize hyperparameters with optuna 
def optimize_hyperparams(img_size=128, epochs=25, limit=100, n_trials=100):
    def objective(trial): 
        latent_dim = trial.suggest_int("latent_dim", 32, img_size) 
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3)
        beta = trial.suggest_float("beta", 0.1, 4.0)

        vae = VAE(latent_dim=latent_dim, img_size=img_size).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

        val_loss = train_vae_gan(img_size=img_size, epochs=epochs, latent_dim=latent_dim, learning_rate=learning_rate, beta=beta, limit=limit)
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
    epochs=20,
    limit=100, 
    n_trials=10
)

print("Completed VAE")
print(f"Best parameters: {best_params}\n Best value: {best_value}\n")

# restore prints 
sys.stdout = sys.__stdout__
