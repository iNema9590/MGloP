import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
# from IF.train_model_mnist import *
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from joblib import Parallel, delayed
import IF
from IF import *
import time
from sklearn.metrics.pairwise import cosine_similarity

# Define the model and load it
model = Net()
model.load_state_dict(torch.load('data/model_mnist.pth', weights_only=True))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define datasets and loaders
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

# Extract images and labels
train_images = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])  # [N_train, 1, 28, 28]
train_embeddings=model(train_images, emd=True).detach().numpy()
train_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
test_images = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])    # [N_test, 1, 28, 28]
test_embeddings=model(test_images, emd=True).detach().numpy()
test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

torch.manual_seed(8)
train_loader = DataLoader(
    train_dataset,
    batch_size=64, shuffle=True, pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1000, shuffle=False, pin_memory=True
)

# Define custom objective
class MulClassObjective(BaseObjective):
    def train_outputs(self, model, batch):
        inputs = batch[0].to(device)
        return model(inputs)

    def train_loss_on_outputs(self, outputs, batch):
        targets = batch[1].to(device)
        return F.cross_entropy(outputs, targets)

    def train_regularization(self, params):
        return L2_WEIGHT * torch.square(params.norm())

    def test_loss(self, model, params, batch):
        inputs, targets = batch[0].to(device), batch[1].to(device)
        outputs = model(inputs)
        return F.cross_entropy(outputs, targets)

# Hyperparameters
L2_WEIGHT = 1e-4

# Initialize the influence module
torch.manual_seed(8)
module = LiSSAInfluenceModule(
    model=model,
    objective=MulClassObjective(),
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    damp=0.001,
    repeat=1,
    depth=1800,
    scale=10,
)

# Prepare indices
# train_idxs = list(range(len(train_loader.dataset)//3))
test_idxs = list(range(len(test_loader.dataset)))
sims=cosine_similarity(test_embeddings, train_embeddings).argsort(axis=1)
def compute_influence(test_idx):
    print(f'Compute influence for test point {test_idx}')
    train_idxs=sims[test_idx][-3000:]
    torch.manual_seed(8)
    influences = module.influences(train_idxs=train_idxs, test_idxs=[test_idx])
    return influences.numpy()

# Parallel computation of influence scores
num_cores = 32

influence_scores = Parallel(n_jobs=num_cores)(delayed(compute_influence)(test_idx) for test_idx in tqdm(test_idxs))
# Step 7: Save the influence scores to a file
influence_scores = np.array(influence_scores)
np.save('data/influence_scores.npy', influence_scores)
print('Influence scores saved to influence_scores.npy')
