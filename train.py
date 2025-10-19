import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from Model import FullModel


def load_data():
    train_data = np.load('./Dataset/FCU/train_data_FCU.npy')
    train_labels = np.load('./Dataset/FCU/train_labels_FCU.npy')
    test_data = np.load('./Dataset/FCU/test_data_FCU.npy')
    test_labels = np.load('./Dataset/FCU/test_labels_FCU.npy')
    return train_data, train_labels, test_data, test_labels

def add_noise(data, noise_factor=0.01):
    noise = np.random.randn(*data.shape) * noise_factor
    return data + noise


class HVACDataset(Dataset):
    def __init__(self, data, labels, augment=False):
        self.data = data
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.augment:
            data = add_noise(data.numpy())
            data = torch.tensor(data, dtype=torch.float32)

        return data, label


def train_model(train_loader, test_loader, model, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs, embeddings = model(X_batch, return_embedding=True)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total


        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                correct_test += (predicted == y_batch).sum().item()
                total_test += y_batch.size(0)

        test_loss /= len(test_loader)
        test_acc = correct_test / total_test

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")


if __name__ == "__main__":

    train_data, train_labels, test_data, test_labels = load_data()

    train_dataset = HVACDataset(train_data, train_labels, augment=True)
    test_dataset = HVACDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    T = 4
    N = train_data.shape[2]
    feature_dim = train_data.shape[3]
    hidden_features = 64
    out_features = 32
    num_heads = 2
    lag_window = 2
    D = 5
    num_classes = 7
    dropout_rate = 0.5
    num_epochs = 50
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

    model = FullModel(
        in_features=feature_dim,
        hidden_features=hidden_features,
        out_features=out_features,
        num_heads=num_heads,
        lag_window=lag_window,
        D=D,
        N=N,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_model(train_loader, test_loader, model, criterion, optimizer, num_epochs=num_epochs, device=device)
