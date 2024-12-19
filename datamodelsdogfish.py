import torch
import numpy as np
import pandas as pd
from IF import *
import os
from sklearn import linear_model
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from sklearn.model_selection import train_test_split
import pickle
from image_explainer import *
# Set device
device = torch.device('cpu')


def fit_model(X, Y):
    C = 1 / (X.shape[0] * L2_WEIGHT)
    sk_clf = linear_model.LogisticRegression(C=C, tol=1e-8, max_iter=1000)
    sk_clf = sk_clf.fit(X.numpy(), Y.numpy())

    # recreate model in PyTorch
    fc = nn.Linear(768, 1, bias=True)
    fc.weight = nn.Parameter(torch.tensor(sk_clf.coef_))
    fc.bias = nn.Parameter(torch.tensor(sk_clf.intercept_))

    pt_clf = nn.Sequential(
        fc,
        nn.Flatten(start_dim=-2),
        nn.Sigmoid()
    )

    return pt_clf.type('torch.FloatTensor')

def generate_sample_sets(X_train, n_sets=1000, fraction=0.3):
    """
    Generate n_sets subsampled sets as proper subsets of X_train.
    """
    n_samples = int(X_train.shape[0] * fraction)
    sample_sets = [torch.tensor(np.random.choice(X_train.shape[0], size=n_samples, replace=False), dtype=torch.long) for _ in range(n_sets)]
    return sample_sets

def train_single_model(train_embeddings, train_labels, sample):
    return fit_model(train_embeddings[sample], train_labels[sample])

def build_dataset(idx, train_embeddings, test_embeddings, models, sample_sets):
    
    dataset = []
    for i, sample in enumerate(sample_sets):
        Ei = torch.zeros(train_embeddings.shape[0], dtype=torch.float32)
        Ei.index_fill_(0, sample, 1)  # Use in-place operation to set values
        p = models[i](test_embeddings[idx].unsqueeze(0))
        Fi = abs(p-clf(test_embeddings[idx].unsqueeze(0)))
        dataset.append((Ei, Fi))
        
    return dataset

def train_lasso_model(dataset):
    """
    Train a Lasso model on the generated dataset and return the parameters.
    """
    X = np.array([pair[0].tolist() for pair in dataset])
    y = np.array([pair[1].item() for pair in dataset])
    
    lasso_model = linear_model.Lasso(alpha=0.005)  # You can adjust the alpha parameter
    lasso_model.fit(X, y)
    return lasso_model.coef_


def save_sample_sets(sample_sets, file_path='sample_setsdogfish.pt'):
    torch.save(sample_sets, file_path)




if __name__ == "__main__":
    L2_WEIGHT = 1e-4


    embeds=torch.load('data/dogfishembeds.pt')

    X_train = torch.tensor(embeds["X_train"])
    Y_train = torch.tensor(embeds["Y_train"])

    X_test = torch.tensor(embeds["X_test"])
    Y_test = torch.tensor(embeds["Y_test"])

    train_set = data.TensorDataset(X_train, Y_train)
    test_set = data.TensorDataset(X_test, Y_test)
    torch.manual_seed(42)
    clf = fit_model(X_train, Y_train)


    clf.eval()  # Set the model to evaluation mode
    # Generate sample sets and save them
    torch.manual_seed(42)
    sample_sets = generate_sample_sets(X_train)
    save_sample_sets(sample_sets)
    # sample_sets = torch.load("sample_setsdogfish.pt")
    torch.manual_seed(42)
    models = Parallel(n_jobs=32)(
        delayed(train_single_model)(X_train, Y_train, sample) for sample in sample_sets
    )    
    with open('data/modelsdog.pkl', 'wb') as file:
        pickle.dump(models, file)
    # with open('data/modelsdog.pkl', 'rb') as file:
    #     models=pickle.load(file)
    # Load pre-trained models
    n_models = len(sample_sets)
    print("Models are saved to a pickle")

   


embeds=[]
for i in tqdm(range(len(X_test))):
    dataset=build_dataset(i, X_train, X_test, models, sample_sets)
    embeds.append(train_lasso_model(dataset))

torch.save(embeds, "data/embeds_DMdog.pt")