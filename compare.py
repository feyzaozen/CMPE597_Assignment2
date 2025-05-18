import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid, save_image
from torchvision.transforms import Resize
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

# Set model and dataset paths, and define global settings like latent dimension and image size.
z_dim = 64
num_samples = 64
image_size = 299
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
project_root = os.path.abspath(os.getcwd())
model_dir = os.path.join(project_root, "assignment 2", "models")
dataset_path = os.path.join(project_root, "assignment 2", "dataset", "train_images.npy")
data_path = os.path.join(project_root, "dataset", "train_images.npy")

# Define the different decoder architectures.
class ConvDecoderLSTM(nn.Module):
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

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 3, 3)
        return self.deconv(x)

class ConvDecoderConv(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 3 * 3)
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

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 3, 3)
        return self.deconv(x)

# Load pretrained decoder weights from disk and set models to evaluation mode.
decoder_lstm = ConvDecoderLSTM().to(device)
decoder_lstm.load_state_dict(torch.load(os.path.join(model_dir, "vae_lstm_conv_decoder.pt"), map_location=device))
decoder_lstm.eval()

decoder_conv = ConvDecoderConv().to(device)
decoder_conv.load_state_dict(torch.load(os.path.join(model_dir, "vae_conv_conv_decoder.pt"), map_location=device))
decoder_conv.eval()

# Sample from standard normal distribution and generate synthetic images.
torch.manual_seed(42)
z = torch.randn(num_samples, z_dim).to(device)
with torch.no_grad():
    samples_lstm = decoder_lstm(z).cpu()
    samples_conv = decoder_conv(z).cpu()

# Display and save a grid of generated images for visual inspection.
def plot_grid(samples, title, filename):
    grid = make_grid(samples, nrow=8, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_grid(samples_lstm, "Generated Samples (LSTM Decoder)", "samples_lstm_grid.png")
plot_grid(samples_conv, "Generated Samples (Conv Decoder)", "samples_conv_grid.png")

# Resize images to 299x299 and repeat across channels for Inception model input.
resize = Resize((image_size, image_size))
def preprocess(tensor):
    tensor = resize(tensor)
    return tensor.repeat(1, 3, 1, 1)

samples_lstm_resized = preprocess(samples_lstm)
samples_conv_resized = preprocess(samples_conv)

# Used as reference distribution for FID score.
real_images_np = np.load(data_path)[:num_samples]
real_images_tensor = torch.tensor(real_images_np / 255.0, dtype=torch.float32).unsqueeze(1)
real_images_resized = preprocess(real_images_tensor)

# Compute Inception Scores for generated images.
is_metric = InceptionScore(normalize=True).to(device)
is_lstm = is_metric(samples_lstm_resized.to(device))
is_conv = is_metric(samples_conv_resized.to(device))

# Compute FID scores between generated and real images.
fid_metric_lstm = FrechetInceptionDistance(normalize=True).to(device)
fid_metric_lstm.update(samples_lstm_resized.to(device), real=False)
fid_metric_lstm.update(real_images_resized.to(device), real=True)
fid_lstm = fid_metric_lstm.compute()

fid_metric_conv = FrechetInceptionDistance(normalize=True).to(device)
fid_metric_conv.update(samples_conv_resized.to(device), real=False)
fid_metric_conv.update(real_images_resized.to(device), real=True)
fid_conv = fid_metric_conv.compute()

# Print the results.
print("\nInception Score (LSTM):", is_lstm)
print("Inception Score (Conv):", is_conv)
print("FID (LSTM vs Real):", fid_lstm.item())
print("FID (Conv vs Real):", fid_conv.item())
