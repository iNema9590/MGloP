import torch
import numpy as np
from IF import *
import os
from sklearn import linear_model
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

# Set device
device = torch.device('cpu')

def generate_sample_sets(X_train, n_sets=1000, fraction=0.3):
    """
    Generate n_sets subsampled sets as proper subsets of X_train.
    """
    n_samples = int(X_train.shape[0] * fraction)
    sample_sets = [torch.tensor(np.random.choice(X_train.shape[0], size=n_samples, replace=False), dtype=torch.long) for _ in range(n_sets)]
    return sample_sets

def train_single_model(train_embeddings, train_labels, sample):
    return fit_model(train_embeddings[sample], train_labels[sample])

def build_dataset_batch(indices, train_embeddings, test_embeddings, models, sample_sets):
    """
    Build datasets for a batch of indices.
    """
    datasets = []
    for idx in indices:
        dataset = []
        for i, sample in enumerate(sample_sets):
            Ei = torch.zeros(train_embeddings.shape[0], dtype=torch.float32)
            Ei.index_fill_(0, sample, 1)  # Use in-place operation to set values
            p = models[i](test_embeddings[idx]).sort()
            Fi = p.values[-1] - p.values[-2]
            dataset.append((Ei, Fi))
        datasets.append(dataset)
    return datasets

def train_lasso_model(dataset):
    """
    Train a Lasso model on the generated dataset and return the parameters.
    """
    X = np.array([pair[0].tolist() for pair in dataset])
    y = np.array([pair[1].item() for pair in dataset])
    
    lasso_model = linear_model.Lasso(alpha=0.005)  # You can adjust the alpha parameter
    lasso_model.fit(X, y)
    return lasso_model.coef_

def save_models(models, directory='models'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i, model in enumerate(models):
        model_path = os.path.join(directory, f'model_{i}.pth')
        torch.save(model.state_dict(), model_path)

def load_models(n_models, model_class=LinearModel, directory='models'):
    models = []
    for i in range(n_models):
        model = model_class()
        model_path = os.path.join(directory, f'model_{i}.pth')
        model.load_state_dict(torch.load(model_path))
        model.to(device)  # Ensure the model is on the correct device
        models.append(model)
    return models

def save_sample_sets(sample_sets, file_path='sample_sets.pt'):
    torch.save(sample_sets, file_path)

# Batch processing for efficiency
def process_embedding_batch(batch_indices):
    """
    Process a batch of indices to train lasso models in parallel.
    """
    batch_results = []
    datasets = build_dataset_batch(batch_indices, train_embeddings_shared, test_embeddings_shared, models_shared, sample_sets_shared)
    for dataset in datasets:
        batch_results.append(train_lasso_model(dataset))
    return batch_results

if __name__ == "__main__":

    model = LinearModel()
    model.load_state_dict(torch.load('data/main_model.pth'))
    model.eval()  # Set the model to evaluation mode

    train_embeddings = torch.load('data/train_embeddings.pt')
    train_labels = torch.load('data/train_labels.pt')
    test_embeddings = torch.load('data/test_embeddings.pt')
    test_labels = torch.load('data/test_labels.pt')
    print('Model and embeddings are loaded.')

    # Generate sample sets and save them
    # sample_sets = generate_sample_sets(train_embeddings)
    # save_sample_sets(sample_sets)
    sample_sets = torch.load("sample_sets.pt")
    # models = Parallel(n_jobs=32)(
    #     delayed(train_single_model)(train_embeddings, train_labels, sample) for sample in sample_sets
    # )    
    # save_models(models)

    # Load pre-trained models
    models = load_models(len(sample_sets), model_class=LinearModel, directory='models')
    n_models = len(sample_sets)
    print("Models are loaded from saved files")

    # Share large immutable data structures globally for efficient parallel processing
    train_embeddings_shared = train_embeddings
    test_embeddings_shared = test_embeddings
    models_shared = models
    sample_sets_shared = sample_sets

    # Parallelize in batches
    batch_size = 10  # Number of test samples per batch
    indices = range(len(test_labels))
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    try:
        embeds_list = Parallel(n_jobs=32, batch_size=1)(
            delayed(process_embedding_batch)(batch) for batch in tqdm(batches)
        )

        # Flatten the results and save
        embeds_flattened = [item for sublist in embeds_list for item in sublist]
        embeds = torch.tensor(embeds_flattened, dtype=torch.float32)
        torch.save(embeds, 'data/embeds_DM.pt')
        print("Proto embeddings saved.")
    finally:
        # Ensure resources are released
        get_reusable_executor().shutdown(wait=True)
