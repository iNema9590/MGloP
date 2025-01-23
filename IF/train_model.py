import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(2048, 10)  # Directly map input to output without hidden layers

    def forward(self, x):
        x = self.fc(x)
        return x
# class LinearModel(nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(2048, 512),  # Add hidden layer
#             nn.ReLU(),
#             nn.Linear(512, 10)
#         )
    
#     def forward(self, x):
#         x = self.fc(x)
#         return x

def fit_model(train_embeddings, train_labels, num_epochs=5, learning_rate=0.001):
    # Create the linear neural network
    model = LinearModel()

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)


    # Prepare data
    train_embeddings = train_embeddings.to(device)
    train_labels = train_labels.to(device)

    # Train the linear neural network
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_embeddings)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print('Finished training')
    return model


def save_embeddings(dataloader, file_prefix, model, device):
    embeddings = []
    labels = []
    with torch.no_grad():
        for data in dataloader:
            images, targets = data
            images = images.to(device)
            output = model(images)
            output = output.view(output.size(0), -1)
            embeddings.append(output.cpu())
            labels.append(targets)
    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)
    torch.save(embeddings, f'{file_prefix}_embeddings2.pt')
    torch.save(labels, f'{file_prefix}_labels2.pt')


if __name__ == "__main__":
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    torch.manual_seed(8)
    trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # # Load a pre-trained ResNet50 model
    resnet = models.resnet50(pretrained=True)

    # # Replace the final fully connected layer with a new one for CIFAR-10
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 10)

    # # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = resnet.to(device)

    # # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=0.001)

    # Fine-tune the ResNet50 model on CIFAR-10
    num_epochs = 5
    for epoch in range(num_epochs):
        resnet.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished fine-tuning')

    # Extract embeddings using the fine-tuned ResNet50 model
    resnet.fc = nn.Identity()  # Remove the classification head to get embeddings

    # Save embeddings
    save_embeddings(trainloader, 'data/train', resnet, device)
    save_embeddings(testloader, 'data/test', resnet, device)
    print("Embeddings are saved")

    # Load the saved embeddings and labels
    train_embeddings = torch.load('./data/train_embeddings2.pt')
    train_labels = torch.load('./data/train_labels2.pt')
    test_embeddings = torch.load('./data/test_embeddings2.pt')
    test_labels = torch.load('./data/test_labels2.pt')

    # Train the linear model on embeddings
    network = fit_model(train_embeddings, train_labels)
    torch.save(network.state_dict(), '../data/main_model2.pth')
    print('Model saved.')

    # Evaluate the model on test data
    network.eval()
    with torch.no_grad():
        outputs = network(test_embeddings.to(device))
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == test_labels.to(device)).sum().item() / len(test_labels)
        print(f'Test Accuracy: {accuracy * 100:.2f}%')
