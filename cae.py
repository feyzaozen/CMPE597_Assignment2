import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
import os

# Seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, embedding_dim=2):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 32, 3, 2, 1)       
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.fc = nn.Linear(128 * 4 * 4, embedding_dim)

    def forward(self, x):
        # First Conv layer: (batch size, 1, 32, 32),(batch size, 32, 16, 16)
        x = F.relu(self.conv1(x))
        # (batch size, 32, 16, 16),(batch size, 64, 8, 8)
        x = F.relu(self.conv2(x))
        # (batch size, 64, 8, 8),(batch size, 128, 4, 4)
        x = F.relu(self.conv3(x))
        # Flatten (batch size, 128*4*4)
        x = x.view(x.size(0), -1)
        # Fully connected (batch size, embedding_dim)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, embedding_dim=2):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, 128 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 3, 2, 1, 1)

    def forward(self, x):
        # Embedding, (batch size, 128, 4, 4)
        x = self.fc(x).view(x.size(0), 128, 4, 4)  
        # (batch size, 128, 4, 4),(batch size, 64, 8, 8)
        x = F.relu(self.deconv1(x))   
        # (batch size, 64, 8, 8),(batch size, 32, 16, 16)             
        x = F.relu(self.deconv2(x))   
        # (batch size, 32, 16, 16),(batch size, 1, 32, 32)             
        return torch.sigmoid(self.deconv3(x))     

# Convolutional autoencoder consisting of encoder and decoder
class ConvAutoencoder(nn.Module):
    def __init__(self, embedding_dim=2):
        super().__init__()
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim)

    def forward(self, x, return_embedding=False):
        # Converts image to latent vector
        z = self.encoder(x)
        # Converts latent vector back to image
        recon = self.decoder(z)
        if return_embedding:
            return recon, z
        return recon

# Model training function
def train_model(model, train_loader, test_loader, num_epochs=20, lr=1e-3):
    # Adam optimizer with learning rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # MSE loss is used
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []

    
    for epoch in range(1, num_epochs+1):
        # Training
        model.train()
        total_tr = 0
        for x, _ in train_loader:
            x = x.to(device)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_tr += loss.item() * x.size(0)
        train_losses.append(total_tr / len(train_loader.dataset))

        # Validation
        model.eval()
        total_val = 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                recon = model(x)
                total_val += criterion(recon, x).item() * x.size(0)
        val_losses.append(total_val / len(test_loader.dataset))

        print(f"Epoch {epoch:02d}: Train MSE={train_losses[-1]:.5f}, Val MSE={val_losses[-1]:.5f}")

    # Loss plot
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.title("Loss vs Epochs"); plt.grid(); plt.legend()
    plt.show()

# t-SNE Visualization
# Returns the encoder output (latent vectors) for all data
def get_all_latents(model, data_tensor, batch_size=64):
    model.eval()
    latents = []
    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            x_batch = data_tensor[i:i+batch_size].to(device)
            _, z = model(x_batch, return_embedding=True)
            latents.append(z.cpu())
    return torch.cat(latents, dim=0)


def visualize_tsne(model, data_tensor, labels, label_map):
    # (N, 2)
    embeddings = get_all_latents(model, data_tensor)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings.numpy())
    # Scatter plot
    plt.figure(figsize=(10, 6))
    for label_id in np.unique(labels):
        idx = labels == label_id
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                    label=label_map[int(label_id)], s=5, alpha=0.7)
    plt.title("t-SNE of CAE Latent Representations")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Classes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    set_seed(42)
    
    # Load data 
    train_images = np.load("dataset/train_images.npy")  
    test_images = np.load("dataset/test_images.npy")    
    train_labels = np.load("dataset/train_labels.npy")
    test_labels = np.load("dataset/test_labels.npy")

    # Pad images from 28x28 to 32x32
    train_images = np.pad(train_images, ((0, 0), (2, 2), (2, 2)), mode='constant')
    test_images = np.pad(test_images, ((0, 0), (2, 2), (2, 2)), mode='constant')

    # Get tensor and normalize
    train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1) / 255.0
    test_images = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1) / 255.0
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)

    # Get dataLoaders
    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=64, shuffle=False)

    # Create Model
    model = ConvAutoencoder(embedding_dim=2).to(device)
    train_model(model, train_loader, test_loader)

    # t-SNE visualization
    label_map = {
        0: "rabbit",
        1: "yoga",
        2: "hand",
        3: "snowman",
        4: "motorbike"
    }
    visualize_tsne(model, train_images, train_labels, label_map)
