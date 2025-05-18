import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from torchvision.transforms import Resize
import numpy as np

#Set latent dimension, number of classes, and device. Define model paths.
z_dim = 64
num_classes = 3
samples_per_class = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = os.path.dirname(os.path.dirname(__file__))
model_dir = os.path.join(base_dir, "assignment 2", "models")
classifier_path = os.path.join(base_dir, "assignment 2", "classifier_a1.pt")

#Combines image and one-hot class vector to produce latent mu and logvar.
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

#Reconstructs image from latent vector and one-hot class label.
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

#Encodes and decodes conditionally using class information.
class ConditionalVAE(nn.Module):
    def __init__(self, z_dim=64, num_classes=3):
        super().__init__()
        self.encoder = ConditionalConvEncoder(z_dim, num_classes)
        self.decoder = ConditionalConvDecoder(z_dim, num_classes)

    def forward(self, x, y_onehot):
        mu, logvar = self.encoder(x, y_onehot)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, y_onehot), mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

#Used to classify generated samples to evaluate generation quality.
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

#Load the classifier model.
classifier = MLP().to(device)
classifier.load_state_dict(torch.load(classifier_path, map_location=device))
classifier.eval()

#Load the Conditional VAE model.
model = ConditionalVAE(z_dim=z_dim, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(os.path.join(model_dir, "cvae_conv_conv_full.pt"), map_location=device))
model.eval()

#Create class-conditional samples and evaluate them using the classifier.
labels_text = ['rabbit', 'yoga', 'snowman']
generated, annotations = [], []

for class_id in range(num_classes):
    label = torch.full((samples_per_class,), class_id, dtype=torch.long).to(device)
    y_onehot = F.one_hot(label, num_classes=num_classes).float()
    z = torch.randn(samples_per_class, z_dim).to(device)

    with torch.no_grad():
        samples = model.decoder(z, y_onehot.to(device)).cpu()

    with torch.no_grad():
        preds = classifier(samples.view(samples.size(0), -1))
        probs = F.softmax(preds, dim=1)
        conf, pred_class = probs.max(dim=1)

    for i in range(samples_per_class):
        true_label = labels_text[class_id]
        predicted = labels_text[pred_class[i]] if pred_class[i] < 3 else f"other({pred_class[i].item()})"
        annotations.append(f"GT: {true_label}\nPred: {predicted}\nConf: {conf[i].item():.2f}")
        print(f"Class: {true_label:<8} | Predicted: {predicted:<10} | Confidence: {conf[i].item():.2f}")

    generated.append(samples)

#Shows the generated images with ground truth, predictions, and confidence scores.
all_samples = torch.cat(generated, dim=0)
fig, axs = plt.subplots(num_classes, samples_per_class, figsize=(samples_per_class * 2, num_classes * 2))

for i, ax in enumerate(axs.flatten()):
    ax.imshow(all_samples[i].squeeze(), cmap='gray')
    ax.set_title(annotations[i], fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(model_dir, "cvae_conv_generated_labeled.png"))
plt.show()
