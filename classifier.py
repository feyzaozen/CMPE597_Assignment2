import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

#Data loader. This part uses the os module to streamline the data loading process. This way, the data loader can work in any machine.
base_dir = os.path.dirname(os.path.dirname(__file__))
dataset_path = os.path.join(base_dir, 'assignment 2/dataset')

train_images = np.load(os.path.join(dataset_path, 'train_images.npy'))
train_labels = np.load(os.path.join(dataset_path, 'train_labels.npy'))
test_images  = np.load(os.path.join(dataset_path, 'test_images.npy'))
test_labels  = np.load(os.path.join(dataset_path, 'test_labels.npy'))

#Here we flatten and normalize the test and train images and apply one-hot encoding conversion to the categories.
#Numpy's flatten() and eye() functions are used to achieve it.
#To normalize the images, we simply divide them by 255 because each grayscale pixel's value ranges from 0 to 255.
X_train = train_images.reshape(len(train_images), -1) / 255.0
X_test  = test_images.reshape(len(test_images), -1)  / 255.0

y_train = torch.tensor(train_labels, dtype=torch.long)
y_test  = torch.tensor(test_labels, dtype=torch.long)

#Here we convert the variables into PyTorch tensors.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)

##This section seperates 10% of the training set for validation.
val_size = int(0.1 * len(X_train_tensor))
X_val_tensor = X_train_tensor[:val_size]
y_val_tensor = y_train[:val_size]
X_train_tensor = X_train_tensor[val_size:]
y_train_tensor = y_train[val_size:]

#Here, we define a class called MLP to handle the forward propagation and define the hidden layers and neurons. The architecture
#is same as the implementation from scratch. ReLU is used again as an activation function.
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#We define the model, optimizer and the loss function. Note that the CrossEntropyLoss function already includes softmaxxing the
#input, so we did not apply softmax again.
model = MLP()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

#Again, same with the first implementation.
epochs = 30
batch_size = 32

#These variables hold the values loss and accuracy values that will be plotted.
train_losses = []
test_accuracies = []
train_accuracies = []

#This section includes the training loop, pretty much the same as the first part.
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    perm = torch.randperm(X_train_tensor.size(0))
    X_train_tensor = X_train_tensor[perm]
    y_train_tensor = y_train_tensor[perm]


    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        #We print the first layer gradients to compare them with the implementation from scratch.
        if epoch == 0 and i == 0:
            print("PyTorch First Layer Gradients (fc1.weight.grad):")
            print(model.fc1.weight.grad)
            
            
            pytorch_grad = model.fc1.weight.grad.detach().numpy().T
            #We print the max difference and mean difference between the gradients of each implementation's first layer.

        
        epoch_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += X_batch.size(0)
    
    avg_loss = epoch_loss / total
    train_acc = correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(train_acc)
    
    model.eval()
    with torch.no_grad():
        outputs_test = model(X_test_tensor)
        _, predicted_test = torch.max(outputs_test, 1)
        test_acc = accuracy_score(y_test.numpy(), predicted_test.numpy())
    test_accuracies.append(test_acc)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

model.eval()
with torch.no_grad():
    outputs_test = model(X_test_tensor)
    _, predicted_test = torch.max(outputs_test, 1)
    final_test_acc = accuracy_score(y_test.numpy(), predicted_test.numpy())
    print(f"Final Test Accuracy: {final_test_acc:.4f}")

#In this section, we printed a class-based version of the model evaluation metrics to assess which classes performed better.
class_names = ['rabbit', 'yoga', 'hand', 'snowman', 'motorbike']
model.eval()
with torch.no_grad():
    outputs_test = model(X_test_tensor)
    _, y_pred_tensor = torch.max(outputs_test, 1)

y_pred = y_pred_tensor.numpy()
y_true = y_test.numpy()

report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).T

print("\n Per-Class Performance (PyTorch Implementation):")
print(report_df[['precision', 'recall', 'f1-score']].round(4))

# === Save the trained model to 'classifier_a1.pt' ===
model_save_path = os.path.join(os.path.dirname(__file__), "classifier_a1.pt")
torch.save(model.state_dict(), model_save_path)
