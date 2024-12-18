import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, emd=False):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        if emd:
            return x
        else:
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim = 1)
    
criterion = nn.CrossEntropyLoss()

def fit_model(train_loader, n_epochs=50):
    # Create the linear neural network
    model = Net()

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # train_losses = []
    # train_counter = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                n_epochs, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        #     train_losses.append(loss.item())
        #     train_counter.append(
        #         (batch_idx*64) + ((n_epochs-1)*len(train_loader.dataset)))
            # torch.save(model.state_dict(), '../data/model_mnist.pth')

    print('Finished training')
    return model


if __name__ == "__main__":

    torch.manual_seed(8)
    train_loader =DataLoader(
    torchvision.datasets.MNIST('../data/', train=True, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=64, shuffle=True)
    torch.manual_seed(8)
    test_loader = DataLoader(
    torchvision.datasets.MNIST('../data/', train=False, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=1000, shuffle=True)

    # # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





    # Train the linear model on embeddings
    torch.manual_seed(8)
    network = fit_model()


    # Evaluate the model on test data
    network.eval()
    test_loss = 0
    correct = 0
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(50 + 1)]
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
