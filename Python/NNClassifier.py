from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import dataCreation as dc


class Network(nn.Module):
    def __init__(self, input_size: int, nr_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, nr_classes)

    def forward(self, x):
        input1 = self.fc1(x)
        hidden = F.relu(input1)
        output = self.fc2(hidden)
        return output


def get_data_loaders(dataset: dc.Data, batch_size: int = 64, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    training, testing = dataset.get_test_training_split()
    training_loader = DataLoader(dataset=training, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=testing, batch_size=batch_size, shuffle=shuffle)
    return training_loader, test_loader


def check_accuracy(loader: DataLoader, model: Network, device: torch.device):
    num_correct = num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct/num_samples) * 100:.2f}")


def train(dataset: dc.Data, lr: float = 0.001, num_epochs: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network(input_size=len(dataset.data_columns), nr_classes=len(dataset.class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader, test_loader = get_data_loaders(dataset=dataset)

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            #forward
            scores = model(data)
            loss = criterion(scores, targets.long())

            #backward
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        print("testing training data")
        check_accuracy(loader=train_loader, model=model, device=device)
        print("testing test data")
        check_accuracy(loader=test_loader, model=model, device=device)




def test_data_generation():
    data = dc.SoccerDataSet(save=False)
    training_loader, test_loader = get_data_loaders(data)
    for idx, (data, target) in enumerate(training_loader):
        print(idx)
        print(data.shape)
        print(target.shape)
        print(data[1, :])
        break



def test():
    dataset = dc.SoccerDataSet(save=False)
    train(dataset=dataset, num_epochs=100)
    #model = Network(784, 10)
    #test_ = torch.randn(64, 784)
    #print(model(test_).shape)


if __name__ == "__main__":
    test()
    #test_data_generation()

