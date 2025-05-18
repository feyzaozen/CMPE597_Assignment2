import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchinfo import summary
import os
import random

# Seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Encoder
# LSTM encoder converts the input sequence into smaller sized representations

class Encoder(nn.Module):
    def __init__(self, input_size=28, hid1_dim=128, hid2_dim=64):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hid1_dim, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hid1_dim, hidden_size=hid2_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        # First LSTM
        # (batch size, 28, 128)
        out, _ = self.lstm1(x)             
        # Second LSTM 
        # Output size: (batch size, 28, 64)
        # h_n size: (1, batch size, 64), final hidden state for each sample
        out, (h_n, _) = self.lstm2(out)     
        # (batch size, 64) embedding
        return h_n[-1]                     


# Decoder
# LSTM decoder converts latent vector back to sequence
class Decoder(nn.Module):
    def __init__(self, output_size=28, hid1_dim=128, hid2_dim=64):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=hid2_dim, hidden_size=hid1_dim, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hid1_dim, hidden_size=output_size, num_layers=1, batch_first=True)

    def forward(self, z, seq_len):
        # seq_len is 28
        # (batch size, seq_len, 64)
        z_seq = z.unsqueeze(1).repeat(1, seq_len, 1)  
        # (batch size, seq_len, 128)
        out, _ = self.lstm1(z_seq) 
        # (batch size, seq_len, output_size)               
        out, _ = self.lstm2(out)                    
        return out


# LSTM based autoencoder consisting of encoder and decoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=28, hid1_dim=128, hid2_dim=64, seq_len=28):
        super().__init__()
        self.encoder = Encoder(input_size, hid1_dim, hid2_dim)
        self.decoder = Decoder(input_size, hid1_dim, hid2_dim)
        self.seq_len = seq_len

    def forward(self, x, return_embedding=False):
        z = self.encoder(x)                    
        recon = self.decoder(z, self.seq_len)   
        if return_embedding:
            return recon, z
        return recon


# Training
def train_model(model, train_loader, test_loader, num_epochs=20, lr=1e-3):
    model = model.to(device)
    # Adam optimizer with learning rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # MSE loss is used
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        total_train_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            recon = model(x)
            loss = criterion(recon, x)

            optimizer.zero_grad()
            # Error backpropagation
            loss.backward()
            # Update weights
            optimizer.step()

            total_train_loss += loss.item() * x.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                recon = model(x)
                loss = criterion(recon, x)
                total_val_loss += loss.item() * x.size(0)

        val_loss = total_val_loss / len(test_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d}: Train MSE={train_loss:.6f}, Val MSE={val_loss:.6f}")

    # Loss plot
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model, train_losses, val_losses



# t-SNE Visualization
# Returns the encoder output (latent vectors) for all data
def get_all_latents(model, data_tensor, batch_size=64):
    model.eval()
    device = next(model.parameters()).device 
    latents = []

    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            x_batch = data_tensor[i:i+batch_size].to(device)
            _, z = model(x_batch, return_embedding=True)  
            latents.append(z.cpu())

    return torch.cat(latents, dim=0)

def visualize_tsne(model, train_x, train_labels, label_map):
    # (N, 64)
    embeddings = get_all_latents(model, train_x) 
    
    # Reducing to two dimension with t-SNE and visualize with scatter plot
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings.numpy())

    # Scatter plot
    plt.figure(figsize=(10, 6))
    for label_id in np.unique(train_labels):
        idx = train_labels == label_id
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                    label=label_map[int(label_id)], s=5, alpha=0.6)
    
    plt.title("t-SNE of Latent Representations")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Classes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    set_seed(42)

    # Load data and normalization
    train_images = np.load('dataset/train_images.npy') 
    test_images  = np.load('dataset/test_images.npy')   

    train_x = torch.tensor(train_images, dtype=torch.float32).unsqueeze(-1) / 255.0  
    test_x  = torch.tensor(test_images,  dtype=torch.float32).unsqueeze(-1) / 255.0

    # batch size, sequence lenght, feature dimension
    train_x = train_x.squeeze(-1)
    test_x  = test_x.squeeze(-1)

    train_ds = TensorDataset(train_x, train_x)
    test_ds  = TensorDataset(test_x,  test_x)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    # Create model
    model = LSTMAutoencoder().to(device)

    # Train
    train_model(model, train_loader, test_loader)

    # t-SNE
    label_map = {
        0: "rabbit",
        1: "yoga",
        2: "hand",
        3: "snowman",
        4: "motorbike"
    }
    train_labels = np.load("dataset/train_labels.npy") 

    visualize_tsne(model, train_x, train_labels, label_map)
