import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IF import *
import numpy as np

# Step 1: Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Step 2: Define a simple neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

# Step 3: Train the model
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(1):  # Train for 5 epochs
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Step 4: Define the influence objective
class MNISTObjective(BaseObjective):
    def train_outputs(self, model, batch):
        data, _ = batch
        return model(data)

    def train_loss_on_outputs(self, outputs, batch):
        _, target = batch
        return F.cross_entropy(outputs, target, reduction='mean')

    def train_regularization(self, params):
        return 0.01 * sum(p.norm() ** 2 for p in params)

    def test_loss(self, model, params, batch):
        data, target = batch
        output = model(data)
        return F.cross_entropy(output, target, reduction='mean')

# Step 5: Initialize the influence module
influence_module = LiSSAInfluenceModule(
                                model=model,
                                objective=MNISTObjective(),
                                train_loader=train_loader,
                                test_loader=test_loader,
                                device="cpu",
                                damp=0.001,
                                repeat= 1,
                                depth=1800,
                                scale= 10,)

# Step 6: Compute influence for all test points
all_test_indices = list(range(len(test_dataset)))
all_train_indices = list(range(len(train_dataset)))

influence_scores = []

# Compute influence for each test point
for test_idx in all_test_indices:
    print(f'Compute influence for test point {test_idx}')
    influences = influence_module.influences(all_train_indices, [test_idx])
    influence_scores.append(influences)
    

# Step 7: Save the influence scores to a file
influence_scores = np.array(influence_scores)
np.save('influence_scores.npy', influence_scores)
print('Influence scores saved to influence_scores.npy')
