import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

#Set model and training parameters. Define paths for dataset and model saving.
z_dim = 64
num_classes = 3  #Classes used: rabbit, yoga, snowman
batch_size = 64
num_epochs = 35
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = os.path.dirname(os.path.dirname(__file__))
dataset_path = os.path.join(base_dir, "assignment 2", "dataset")
model_path = os.path.join(base_dir, "assignment 2", "models", "cvae_conv_conv_full.pt")

#Only include the specified classes and normalize pixel values.
X = np.load(os.path.join(dataset_path, "train_images.npy"))
y = np.load(os.path.join(dataset_path, "train_labels.npy"))
mask = y < 3  #Filter out all but the three selected classes
X = X[mask]
y = y[mask]

X = X / 255.0  #Normalize to [0, 1]
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#Takes both image and class condition and outputs latent mean and log-variance.
class ConditionalConvEncoder(nn.Module):
    def __init__(self, latent_dim=64, num_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1 + num_classes, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x, y_onehot):
        y = y_onehot.view(y_onehot.size(0), num_classes, 1, 1).expand(-1, -1, 28, 28)
        x = torch.cat([x, y], dim=1)
        h = self.conv(x)
        h = self.flatten(h)
        return self.fc_mu(h), self.fc_logvar(h)

#Reconstructs images from latent vector and class condition.
class ConditionalConvDecoder(nn.Module):
    def __init__(self, latent_dim=64, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim + num_classes, 256 * 3 * 3)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 5, stride=1),
            nn.Sigmoid()
        )

    def forward(self, z, y_onehot):
        z = torch.cat([z, y_onehot], dim=1)
        x = self.fc(z).view(-1, 256, 3, 3)
        return self.deconv(x)

#Combines encoder, reparameterization trick, and decoder into a full model.
class ConditionalVAE(nn.Module):
    def __init__(self, z_dim=64, num_classes=3):
        super().__init__()
        self.encoder = ConditionalConvEncoder(z_dim, num_classes)
        self.decoder = ConditionalConvDecoder(z_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y_onehot):
        mu, logvar = self.encoder(x, y_onehot)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, y_onehot)
        return recon, mu, logvar

#Combines reconstruction loss with KL divergence regularization.
def compute_losses(recon, x, mu, logvar):
    rec_loss = F.mse_loss(recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return rec_loss, kl

#Uses KL annealing to slowly introduce regularization. Trains over multiple epochs.
model = ConditionalVAE(z_dim=z_dim, num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()
    total_rec, total_kl = 0, 0
    beta = min(0.4, epoch / (num_epochs - 1))

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_onehot = F.one_hot(y_batch, num_classes=num_classes).float().to(device)

        recon, mu, logvar = model(x_batch, y_onehot)
        rec_loss, kl = compute_losses(recon, x_batch, mu, logvar)
        loss = rec_loss + beta * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_rec += rec_loss.item()
        total_kl += kl.item()

    print(f"Epoch {epoch+1} | Recon Loss: {total_rec:.2f} | KL: {total_kl:.2f} | β={beta:.2f}")

#Saves the trained model weights for future inference or evaluation.
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
