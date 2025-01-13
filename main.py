import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# model  = NN(784, 10)
# x = torch.randn(64, 784)
# print(model(x).shape) # its going to be 64 x 10

# Set deice... here we gonna do on CPU.
device = torch.device('cpu')

# Hyperparameters
input_size = 784
num_classes = 10
lerning_rate = 0.001
batch_sie = 64
num_epochs = 1

# Dataset...
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_sie, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=train_dataset, batch_size=batch_sie, shuffle=True)

#init model
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#Loss and Optimiser
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=lerning_rate)

# Train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Get to correct shape..
        data = data.reshape(data.shape[0], -1)
        # Forward
        source = model(data)
        loss = criterion(source, targets)
        # backward
        optimiser.zero_grad()
        loss.backward()
        # Gradient decent or adam step...
        optimiser.step()

# Check for the acccuricy...
def check_accuricy(loader, model):
    if loader.dataset.train:
        print("Checking accuricy on tranining data")
    else:
        print("Checking accuricy on test data.")
    num_samples = 0
    num_correct = 0
    model.eval()

    with torch.no_grad():
        for x, y  in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0], -1)
            source = model(x)
            _, prediction = source.max(1)
            num_correct+=(prediction == y).sum()
            num_samples+=prediction.size(0)
        print(f'Got {num_correct} / {num_samples} with acccuricy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()


check_accuricy(train_loader, model)
check_accuricy(test_loader, model)
