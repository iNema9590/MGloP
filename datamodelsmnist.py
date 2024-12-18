import torch
import torch.nn as nn
import numpy as np
import os
from sklearn import linear_model
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from torchvision import datasets, transforms
from IF.train_model_mnist import *  # Ensure that Net, fit_model, and LinearModel are defined here
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cpu')

def generate_sample_sets(n_train_samples, n_sets=1000, fraction=0.3):
    """
    Generate n_sets subsampled sets as proper subsets of the training indices.
    """
    n_samples = int(n_train_samples * fraction)
    sample_sets = [torch.tensor(np.random.choice(n_train_samples, size=n_samples, replace=False), dtype=torch.long) for _ in range(n_sets)]
    return sample_sets

def train_single_model(train_images, train_labels, sample):
    # Flatten the images if necessary
    train_data = train_images[sample]
    train_targets = train_labels[sample]

    # Create a TensorDataset and then a DataLoader
    train_dataset = TensorDataset(train_data, train_targets)
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Now trainloader can be passed to fit_model directly
    return fit_model(trainloader)

def build_dataset_batch(indices, train_images, test_images, models, sample_sets):
    """
    Build datasets for a batch of indices.
    Ei is an indicator vector for which training samples were in the subset.
    Fi is derived from the model predictions on the test sample.
    """
    datasets = []
    n_train = len(train_images)
    for idx in indices:
        dataset = []
        # Flatten test image if needed
        test_input = test_images[idx]  # shape [1, 784]
        for i, sample in enumerate(sample_sets):
            Ei = torch.zeros(n_train, dtype=torch.float32)
            Ei.index_fill_(0, sample, 1)  # Mark these indices as selected
            
            # Model inference
            p = models[i](test_input).sort()
            Fi = p.values[0, -1] - p.values[0, -2]  # difference between top 2 values
            dataset.append((Ei, Fi))
        datasets.append(dataset)
    return datasets

def train_lasso_model(dataset):
    """
    Train a Lasso model on the generated dataset and return the parameters.
    """
    X = np.array([pair[0].tolist() for pair in dataset])
    y = np.array([pair[1].item() for pair in dataset])
    
    lasso_model = linear_model.Lasso(alpha=0.005)  # Adjust alpha as needed
    lasso_model.fit(X, y)
    return lasso_model.coef_

def save_models(models, directory='modelsmnist'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i, model in enumerate(models):
        model_path = os.path.join(directory, f'model_{i}.pth')
        torch.save(model.state_dict(), model_path)

def load_models(n_models, model_class=Net, directory='modelsmnist'):
    models = []
    for i in range(n_models):
        model = model_class()
        model_path = os.path.join(directory, f'model_{i}.pth')
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        models.append(model)
    return models

def save_sample_sets(sample_sets, file_path='sample_setsmnist.pt'):
    torch.save(sample_sets, file_path)

def process_embedding_batch(batch_indices):
    """
    Process a batch of indices to train lasso models in parallel.
    """
    batch_results = []
    datasets = build_dataset_batch(batch_indices, train_images_shared, test_images_shared, models_shared, sample_sets_shared)
    for dataset in datasets:
        batch_results.append(train_lasso_model(dataset))
    return batch_results

if __name__ == "__main__":
    # Load the MNIST model
    model = Net()
    model.load_state_dict(torch.load('data/model_mnist.pth', map_location='cpu', weights_only=True))
    model.eval()
    model.share_memory()
    print("Model Loaded")

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    # Extract images and labels
    train_images = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])  # [N_train, 1, 28, 28]
    train_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    test_images = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])    # [N_test, 1, 28, 28]
    test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    print('MNIST data loaded.')

    # Generate sample sets and save them
    sample_sets = generate_sample_sets(len(train_images))
    save_sample_sets(sample_sets)
    sample_sets = torch.load("sample_setsmnist.pt")
    print("samplesets are saved")
    # Train multiple models on subsets of the training data
    models = Parallel(n_jobs=32)(
        delayed(train_single_model)(train_images, train_labels, sample) for sample in sample_sets
    )
    save_models(models)

    # Load pre-trained models
    models = load_models(len(sample_sets), model_class=Net, directory='modelsmnist')
    n_models = len(sample_sets)
    print("Models are loaded from saved files")

    # Share data globally for parallel processing
    train_images_shared = train_images
    test_images_shared = test_images
    models_shared = models
    sample_sets_shared = sample_sets

    # Parallelize in batches
    batch_size = 10
    indices = range(len(test_labels))
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    try:
        embeds_list = Parallel(n_jobs=32, batch_size=50)(
            delayed(process_embedding_batch)(batch) for batch in tqdm(batches)
        )

        # Flatten the results and save
        embeds_flattened = [item for sublist in embeds_list for item in sublist]
        embeds = torch.tensor(embeds_flattened, dtype=torch.float32)
        torch.save(embeds, 'data/embedsmnist_DM.pt')
        print("Proto embeddings saved.")
    finally:
        get_reusable_executor().shutdown(wait=True)
