import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchinfo import summary

# Load grayscale image data (normalized to [-1, 1]), convert to tensors, and prepare DataLoader.
base_dir = os.path.dirname(os.path.dirname(__file__))
dataset_path = os.path.join(base_dir, 'assignment 2/dataset')
train_images = (np.load(os.path.join(dataset_path, 'train_images.npy')) / 255.0 - 0.5) / 0.5
test_images = (np.load(os.path.join(dataset_path, 'test_images.npy')) / 255.0 - 0.5) / 0.5

train_tensor = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1)
test_tensor = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1)
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=128, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set latent dimension for the VAE.
z_dim = 64

# Encode a 28x28 image into a latent vector using an LSTM over the image rows.
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=128, latent_dim=z_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self._init_weights()

    def _init_weights(self):
        # Initialize weights with He for LSTM and Xavier for linear layers
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.xavier_normal_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, x):
        x = x.squeeze(1)
        _, (h_n, _) = self.lstm(x)
        h = h_n.squeeze(0)
        return self.mu(h), self.logvar(h)

# Decodes latent vector into 28x28 image using transpose convolutions.
class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 3 * 3),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, 5, stride=1),
            nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        # Apply He and Xavier initializations
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 3, 3)
        return self.deconv(x)

# Combines LSTMEncoder and ConvDecoder using reparameterization.
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LSTMEncoder()
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

# Computes VAE loss: reconstruction loss + KL divergence.
def compute_losses(recon, x, mu, logvar):
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl

# Initialize model and optimizer
model = VAE().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Print model summaries
print("Encoder Summary:")
summary(model.encoder, input_size=(1, 1, 28, 28))
print("Decoder Summary:")
summary(model.decoder, input_size=(1, z_dim))

#train the model.
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

#plot the loss curves.
plt.figure(figsize=(8, 5))
plt.plot(rec_losses, label="Reconstruction Loss")
plt.plot(kl_losses, label="KL Divergence")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE (LSTM Encoder + Conv Decoder) - Improved")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("vae_lstm_conv_loss_plot.png")
plt.show()

# Show 5 input images and their reconstructions
model.eval()
with torch.no_grad():
    x = train_tensor[:5].to(device)
    recon, _, _ = model(x)
    recon = recon.cpu()

for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x[i].cpu().squeeze().clamp(0, 1), cmap='gray')
    plt.axis('off')
    plt.subplot(2, 5, 5 + i + 1)
    plt.imshow(recon[i].squeeze().clamp(0, 1), cmap='gray')
    plt.axis('off')

plt.suptitle("Original (Top) vs Reconstructed (Bottom)")
plt.tight_layout()
plt.savefig("vae_lstm_conv_recon_grid.png")
plt.show()

#Save the model.
model_dir = os.path.join(base_dir, "assignment 2/models")
os.makedirs(model_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_dir, "vae_lstm_conv_full.pt"))
torch.save(model.decoder.state_dict(), os.path.join(model_dir, "vae_lstm_conv_decoder.pt"))
