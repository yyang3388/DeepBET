import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Define the model
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 150),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(150, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature.clone().detach().float() if isinstance(feature, torch.Tensor) else torch.tensor(feature, dtype=torch.float32)
        self.label = label.clone().detach().float() if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.float32)

    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    def __len__(self):
        return len(self.label)

# Function to train and validate model
def train_model(model, dataloader_train, dataloader_val, device, epochs=15, lr=0.005):
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for feature, label in dataloader_train:
            feature, label = feature.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(feature).squeeze()
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(dataloader_train)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for feature, label in dataloader_val:
                feature, label = feature.to(device), label.to(device)

                output = model(feature).squeeze()
                loss = loss_fn(output, label)

                val_loss += loss.item()

        val_loss /= len(dataloader_val)
        val_losses.append(val_loss)

    return model, train_losses, val_losses

# Main function
def do_train(Y, X, Z, device="cpu"):
    # Train-test split
    z_train, z_test, y_train, y_test, x_train, x_test = train_test_split(
        Z, Y, X, test_size=0.8, random_state=42
    )

    # Data loaders
    dataloader_Y = DataLoader(CustomDataset(z_train, y_train), batch_size=50, shuffle=True)
    dataloader_Y_val = DataLoader(CustomDataset(z_test, y_test), batch_size=50, shuffle=False)
    dataloader_X = DataLoader(CustomDataset(z_train, x_train), batch_size=50, shuffle=True)
    dataloader_X_val = DataLoader(CustomDataset(z_test, x_test), batch_size=50, shuffle=False)

    # Train Y model
    model_Y = LinearModel()
    model_Y, train_loss_Y, val_loss_Y = train_model(model_Y, dataloader_Y, dataloader_Y_val, device)

    # Train X model
    model_X = LinearModel()
    model_X, train_loss_X, val_loss_X = train_model(model_X, dataloader_X, dataloader_X_val, device)

    # Get final outputs
    model_Y.eval()
    model_X.eval()
    with torch.no_grad():
        z_test_tensor = z_test.clone().detach().float().to(device) if isinstance(z_test, torch.Tensor) else torch.tensor(z_test, dtype=torch.float32, device=device)
        y_output = model_Y(z_test_tensor).squeeze().cpu().numpy()
        x_output = model_X(z_test_tensor).squeeze().cpu().numpy()

    return y_output, x_output, np.array(y_test), np.array(x_test)