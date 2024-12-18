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
# Set device
device = torch.device('cpu')
def fit_model(X, Y):
    # Ensure input size matches model definition
    input_size = X.shape[1]  # Number of features
    C = 1 / (X.shape[0] * L2_WEIGHT)
    
    # Train logistic regression in sklearn
    sk_clf = linear_model.LogisticRegression(C=C, tol=1e-8, max_iter=1000)
    sk_clf = sk_clf.fit(X.numpy(), Y.numpy())

    # Recreate PyTorch model
    fc = nn.Linear(input_size, 1, bias=True)
    fc.weight = nn.Parameter(torch.tensor(sk_clf.coef_, dtype=torch.float32))
    fc.bias = nn.Parameter(torch.tensor(sk_clf.intercept_, dtype=torch.float32))

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
        Fi = abs(p-model(test_embeddings[idx].unsqueeze(0)))
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


def save_sample_sets(sample_sets, file_path='sample_setstxt.pt'):
    torch.save(sample_sets, file_path)




if __name__ == "__main__":
    L2_WEIGHT = 1e-4


    df=pd.read_pickle("data/spambert1.pkl")
    X=torch.stack(df.embedding.tolist())
    y=df.label.map({'spam':1, 'ham':0})

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.4, random_state=30)
    text_train = df.message[Y_train.index].tolist()
    X_train = torch.tensor(X_train)
    Y_train = torch.tensor(Y_train.tolist()).type('torch.FloatTensor')

    text_test = df.message[Y_test.index].tolist()
    Y_test = torch.tensor(Y_test.tolist()).type('torch.FloatTensor')
    X_test = torch.tensor(X_test)
    print('Model and embeddings are loaded.')
    model=fit_model(X_train, Y_train)
    model.eval()  # Set the model to evaluation mode
    # Generate sample sets and save them
    # sample_sets = generate_sample_sets(X_train)
    # save_sample_sets(sample_sets)
    sample_sets = torch.load("sample_setstxt.pt")
    # models = Parallel(n_jobs=32)(
    #     delayed(train_single_model)(X_train, Y_train, sample) for sample in sample_sets
    # )    
    # with open('data/models.pkl', 'wb') as file:
    #     pickle.dump(models, file)
    with open('data/models.pkl', 'rb') as file:
        models=pickle.load(file)
    # Load pre-trained models
    n_models = len(sample_sets)
    print("Models are saved to a pickle")

   


embeds=[]
for i in tqdm(range(len(X_test))):
    dataset=build_dataset(i, X_train, X_test, models, sample_sets)
    embeds.append(train_lasso_model(dataset))

torch.save(embeds, "data/embeds_DMtxt.pt")