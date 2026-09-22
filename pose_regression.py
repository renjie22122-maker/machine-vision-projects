import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class PoseDataset(Dataset):

    def __init__(self, feature_dir, augment=False):
        self.feature_dir = feature_dir
        self.files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
        self.labels = [int(f.split('_')[0]) for f in self.files]
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.feature_dir, self.files[idx])
        features = np.load(path)
        label = self.labels[idx]
        if self.augment:
            noise = np.random.normal(0, 0.02, features.shape).astype(np.float32)
            features = features + noise
            if np.random.rand() > 0.5:
                shift = np.random.randint(-10, 10)
                features = np.roll(features, shift, axis=0)
        return (torch.from_numpy(features), label)

def get_dataloaders(feature_dir, batch_size=8, val_split=0.2):
    full_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
    full_labels = [int(f.split('_')[0]) for f in full_files]
    indices = list(range(len(full_files)))
    np.random.shuffle(indices)
    split = int(np.floor(val_split * len(full_files)))
    train_idx, val_idx = (indices[split:], indices[:split])
    train_dataset = PoseDataset(feature_dir, augment=True)
    val_dataset = PoseDataset(feature_dir, augment=False)
    train_sub = torch.utils.data.Subset(train_dataset, train_idx)
    val_sub = torch.utils.data.Subset(val_dataset, val_idx)
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
    return (train_loader, val_loader)

class PoseLSTMRegressor(nn.Module):

    def __init__(self, input_dim=51, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.0)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 1))

    def forward(self, x):
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        avg_pool = torch.mean(lstm_out, dim=1)
        return self.fc(avg_pool)

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets_float = targets.float().unsqueeze(1).to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets_float)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predicted = torch.round(outputs).squeeze(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return (total_loss / len(train_loader), correct / total)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets_float = targets.float().unsqueeze(1).to(device)
            targets_long = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets_float).item()
            total_loss += loss
            predicted = torch.round(outputs).squeeze(1)
            correct += predicted.eq(targets_long).sum().item()
            total += targets.size(0)
    return (total_loss / len(test_loader), correct / total)
