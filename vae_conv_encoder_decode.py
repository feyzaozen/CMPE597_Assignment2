import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torchinfo import summary
from torchvision.utils import make_grid

# Load grayscale images and normalize them to [0, 1]. Convert to tensors and create DataLoader.
project_root = os.path.abspath(os.path.join(os.getcwd()))
dataset_path = os.path.join(project_root, 'dataset')
train_images = np.load(os.path.join(dataset_path, 'train_images.npy')) / 255.0
test_images = np.load(os.path.join(dataset_path, 'test_images.npy')) / 255.0

train_tensor = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1)
test_tensor = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(train_tensor), batch_size=128, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set latent dimension for the VAE.
z_dim = 64

# Maps a 28x28 image to a latent vector using Conv layers and linear layers for μ and logσ².
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=z_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # → (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # → (64, 7, 7)
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc_mu(x), self.fc_logvar(x)

# Decodes latent vector to image using a fully connected layer and ConvTranspose layers.
class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=z_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 3 * 3)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # (128, 6, 6)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # (64, 12, 12)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # (32, 24, 24)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 5, stride=1),                # (1, 28, 28)
            nn.Sigmoid()  # Output normalized to [0, 1]
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 3, 3)
        return self.deconv(x)

# Combines encoder and decoder and applies the reparameterization.
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# Computes total loss as the sum of reconstruction error and KL divergence.
def compute_losses(recon, x, mu, logvar):
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl

#Initialize the model and optimizer.
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Display model summaries
print("Encoder Summary:")
summary(model.encoder, input_size=(1, 1, 28, 28))
print("Decoder Summary:")
summary(model.decoder, input_size=(1, z_dim))

#Model training.
num_epochs = 35
losses, rec_losses, kl_losses = [], [], []

for epoch in range(num_epochs):
    model.train()
    total_loss, total_rec, total_kl = 0, 0, 0
    beta = min(0.4, epoch / (num_epochs - 1))

    for batch in train_loader:
        imgs = batch[0].to(device)
        recon, mu, logvar = model(imgs)
        rec_loss, kl = compute_losses(recon, imgs, mu, logvar)
        loss = rec_loss + beta * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_rec += rec_loss.item()
        total_kl += kl.item()

    losses.append(total_loss)
    rec_losses.append(total_rec)
    kl_losses.append(total_kl)
    print(f"Epoch {epoch+1} | β={beta:.2f} | Total: {total_loss:.2f} | Recon: {total_rec:.2f} | KL: {total_kl:.2f}")

# Plot the losses.
plt.figure(figsize=(8, 5))
plt.plot(rec_losses, label="Reconstruction Loss")
plt.plot(kl_losses, label="KL Divergence")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE (Conv Encoder + Conv Decoder)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("vae_conv_conv_loss_plot.png")
plt.show()

# Compare original and reconstructed images side by side.
model.eval()
with torch.no_grad():
    x = train_tensor[:5].to(device)
    recon, _, _ = model(x)
    recon = recon.cpu()

for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x[i].cpu().squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 5, 5 + i + 1)
    plt.imshow(recon[i].squeeze(), cmap='gray')
    plt.axis('off')

plt.suptitle("Original (Top) vs Reconstructed (Bottom)")
plt.tight_layout()
plt.savefig("vae_conv_conv_recon_grid.png")
plt.show()

# Save the model.
model_dir = os.path.join(project_root, "assignment 2", "models")
os.makedirs(model_dir, exist_ok=True)

full_model_path = os.path.join(model_dir, "vae_conv_conv_full.pt")
decoder_path = os.path.join(model_dir, "vae_conv_conv_decoder.pt")

torch.save(model.state_dict(), full_model_path)
torch.save(model.decoder.state_dict(), decoder_path)
