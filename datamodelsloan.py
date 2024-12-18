import torch
import numpy as np
import pandas as pd
from IF import *
import os
import torch.optim as optim
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn import linear_model

device = torch.device('cpu')
class LoanApprovalNN(nn.Module):
    def __init__(self, input_size):
        super(LoanApprovalNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, 16)
        self.hidden2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, return_embedding=False):
        x = self.relu(self.hidden1(x))
        embedding = self.relu(self.hidden2(x))
        if return_embedding:
            return embedding
        x = self.sigmoid(self.output(embedding))
        return x

# Fit model function
def fit_model(X, y, emb=False):
    model = LoanApprovalNN(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop
    epochs = 200
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    
    if emb:
        model.eval()
        with torch.no_grad():
            embeddings = model(X, return_embedding=True)
        return embeddings
    return model


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
        Fi = abs(p-network(test_embeddings[idx].unsqueeze(0)))
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


    df = pd.read_csv('data/loan_approval_dataset.csv')
    df=df.drop('loan_id',axis=1)
    df[' education'] = LabelEncoder().fit_transform(df[' education'])
    df[' self_employed'] = LabelEncoder().fit_transform(df[' self_employed'])
    df[' loan_status'] = LabelEncoder().fit_transform(df[' loan_status'])
    y = df[' loan_status']

    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop(columns=[' loan_status']))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    train_set = data.TensorDataset(X_train, y_train)
    test_set = data.TensorDataset(X_test, y_test)
    network=LoanApprovalNN(X_train.shape[1])
    network.load_state_dict(torch.load('data/main_loan.pth', map_location='cpu', weights_only=True))
    network.eval()
    torch.manual_seed(8)
    sample_sets = generate_sample_sets(X_train)
    # save_sample_sets(sample_sets)
    # sample_sets = torch.load("sample_setstxt.pt")
    # models = Parallel(n_jobs=32)(
    #     delayed(train_single_model)(X_train, y_train, sample) for sample in sample_sets
    # )    
    # with open('data/models_loan.pkl', 'wb') as file:
    #     pickle.dump(models, file)
    with open('data/models_loan.pkl', 'rb') as file:
        models=pickle.load(file)
    # Load pre-trained models
    n_models = len(sample_sets)
    print("Models are saved to a pickle")

   


    embeds=[]
    for i in tqdm(range(len(X_test))):
        dataset=build_dataset(i, X_train, X_test, models, sample_sets)
        embeds.append(train_lasso_model(dataset))

    torch.save(embeds, "data/embeds_DMloan.pt")