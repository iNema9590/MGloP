import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from IF import *
import torch.nn.functional as F
import time
from joblib import Parallel, delayed
import IF
from IF import *

# Load the model
model = LinearModel()
model.load_state_dict(torch.load('data/main_model.pth', weights_only=True))
model.share_memory()  # Share memory for multiprocessing
print("Model Loaded")

# Load the data
train_embeddings = torch.load('data/train_embeddings.pt', weights_only=True)
train_labels = torch.load('data/train_labels.pt', weights_only=True)
test_embeddings = torch.load('data/test_embeddings.pt', weights_only=True)
test_labels = torch.load('data/test_labels.pt', weights_only=True)

# Create TensorDataset for train and test data
traindataset = TensorDataset(train_embeddings, train_labels)
testdataset = TensorDataset(test_embeddings, test_labels)

# Create DataLoader for train and test data
batch_size = 100
torch.manual_seed(8)
train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)

class MulClassObjective(IF.BaseObjective):

    def train_outputs(self, model, batch):
        return model(batch[0])

    def train_loss_on_outputs(self, outputs, batch):
        return F.cross_entropy(outputs, batch[1])

    def train_regularization(self, params):
        return L2_WEIGHT * torch.square(params.norm())

    def test_loss(self, model, params, batch):
        outputs = model(batch[0])
        return F.cross_entropy(outputs, batch[1])

L2_WEIGHT = 1e-4

module = LiSSAInfluenceModule(
    model=model,
    objective=MulClassObjective(),
    train_loader=train_loader,
    test_loader=test_loader,
    device='cpu',
    damp=0.001,
    repeat=1,
    depth=1800,
    scale=10,
)

train_idxs = list(range(len(train_labels)))

# Load the chunk of test indices
test_idxs = list(range(len(test_labels)))


# def compute_influence(test_idx):
#     print(f'Compute influence for test point {test_idx}')
#     torch.manual_seed(8)
#     influences = module.influences(train_idxs=train_idxs, test_idxs=[test_idx])
#     return influences.numpy()

# # Parallel computation of influence scores
# num_cores = 32

# influence_scores = Parallel(n_jobs=num_cores)(delayed(compute_influence)(test_idx) for test_idx in tqdm(test_idxs))
# # Step 7: Save the influence scores to a file
# influence_scores = np.array(influence_scores)
# np.save('data/influence_scores.npy', influence_scores)
# print('Influence scores saved to influence_scores.npy')

start_time = time.time()
print("start computation of influence")
module.influences(train_idxs=train_idxs[:25000], test_idxs=[1])
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")