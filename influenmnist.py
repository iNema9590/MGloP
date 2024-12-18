import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from IF.train_model_mnist import *
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from joblib import Parallel, delayed
import IF
from IF import *
# Load the model
model = Net()
model.load_state_dict(torch.load('data/model_mnist.pth', weights_only=False))  # Use map_location for compatibility
model.eval()
print("Model Loaded")

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move model to GPU/CPU

# Load datasets with pinned memory for GPU
torch.manual_seed(8)
train_loader = DataLoader(
    datasets.MNIST(
        'data/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    ),
    batch_size=64, shuffle=True, pin_memory=True
)
test_loader = DataLoader(
    datasets.MNIST(
        'data/', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    ),
    batch_size=1000, shuffle=False, pin_memory=True
)

class MulClassObjective(IF.BaseObjective):
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

L2_WEIGHT = 1e-4

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
module._compute_test_loss_gradient(test_idx)
# Convert train and test indices to lists
train_idxs = list(range(len(train_loader.dataset)))
test_idxs = list(range(len(test_loader.dataset)))

# Batch processing for test indices
def compute_influence_batch(test_idx_batch):
    torch.manual_seed(8)
    influences = module.influences(train_idxs=train_idxs, test_idxs=[test_idx_batch])
    return influences.numpy()

# Parallel computation of influence scores
influence_scores = Parallel(n_jobs=-1)(
    delayed(compute_influence_batch) (i) for i in tqdm(test_idxs)
)

# Flatten the results and save
influence_scores = np.vstack(influence_scores)
np.save('data/influence_scores_mnist.npy', influence_scores)