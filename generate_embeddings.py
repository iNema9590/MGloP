import torch
from IF import *
import numpy as np
from sklearn_extra.cluster import KMedoids
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from tqdm import tqdm



model = LinearModel()
model.load_state_dict(torch.load('data/main_model.pth'))
model.eval()  # Set the model to evaluation mode

train_embeddings = torch.load('data/train_embeddings.pt')
train_labels = torch.load('data/train_labels.pt')
test_embeddings = torch.load('data/test_embeddings.pt')
test_labels = torch.load('data/test_labels.pt')

def find_prototypes(embeddings, labels, N, method='per_class_variance'):

    # Convert embeddings and labels to numpy arrays for use with scikit-learn
    embeddings_np = embeddings.cpu().detach().numpy()
    labels_np = labels.cpu().detach().numpy()
    
    if method == 'per_class':
        # Find prototypes for each class equally
        unique_classes = np.unique(labels_np)
        n_classes = len(unique_classes)
        prototypes_per_class = max(1, N // n_classes)
        
        def find_class_prototypes(cls):
            class_indices = np.where(labels_np == cls)[0]
            class_embeddings = embeddings_np[class_indices]
            
            # Determine number of clusters (prototypes_per_class) per class
            n_clusters = min(prototypes_per_class, len(class_embeddings))
            
            kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, method='pam')
            kmedoids.fit(class_embeddings)
            
            # Get the medoid indices relative to the class embeddings
            medoid_indices = class_indices[kmedoids.medoid_indices_]
            return medoid_indices
        
        # Parallelize the prototype finding for each class
        results = Parallel(n_jobs=-1)(delayed(find_class_prototypes)(cls) for cls in unique_classes)
        
        # Collect the results
        prototype_indices = np.hstack(results)
    
    elif method == 'per_class_variance':
        # Find prototypes for each class based on variance
        unique_classes = np.unique(labels_np)
        total_variance = sum(np.var(embeddings_np[np.where(labels_np == cls)[0]], axis=0).mean() for cls in unique_classes)
        
        def find_class_prototypes(cls):
            class_indices = np.where(labels_np == cls)[0]
            class_embeddings = embeddings_np[class_indices]
            
            # Calculate variance for the class
            variance = np.var(class_embeddings, axis=0).mean()
            
            # Determine number of clusters based on variance proportionally (each class must have at least 1 prototype)
            n_clusters = max(1, int((variance / total_variance) * N))
            n_clusters = min(n_clusters, len(class_embeddings))
            
            kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, method='pam')
            kmedoids.fit(class_embeddings)
            
            # Get the medoid indices relative to the class embeddings
            medoid_indices = class_indices[kmedoids.medoid_indices_]
            return medoid_indices
        
        # Parallelize the prototype finding for each class
        results = Parallel(n_jobs=-1)(delayed(find_class_prototypes)(cls) for cls in unique_classes)
        
        # Collect the results
        prototype_indices = np.hstack(results)
    
    elif method == 'global':
        # Find prototypes without considering class labels
        n_clusters = min(N, len(embeddings_np))
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, method='pam')
        kmedoids.fit(embeddings_np)
        prototype_indices = kmedoids.medoid_indices_
    
    else:
        raise ValueError("Invalid method specified. Choose from 'per_class', 'per_class_variance', or 'global'.")
    
    return prototype_indices

def nearest_medoid_accuracy(embeddings, labels, prototype_embeddings, prototype_labels, model):

    # Convert embeddings to numpy array
    embeddings_np = embeddings.cpu().detach().numpy()
    prototype_embeddings_np = prototype_embeddings.cpu().detach().numpy()
    
    # Compute distances from each embedding to each prototype
    distances = pairwise_distances(embeddings_np, prototype_embeddings_np, metric='euclidean')
    nearest_medoid_indices = np.argmin(distances, axis=1)
    predicted_labels = prototype_labels[nearest_medoid_indices]
    
    # Get true labels
    true_labels = labels.cpu().detach().numpy()
    
    # Compute accuracy of the nearest medoid classifier
    nearest_medoid_acc = accuracy_score(true_labels, predicted_labels)
    
    return nearest_medoid_acc

def compute_prototype_silhouette_score(embeddings, prototype_indices):

    # Convert embeddings to numpy arrays
    embeddings_np = embeddings.cpu().detach().numpy()
    prototype_embeddings_np = embeddings_np[prototype_indices]
    
    # Assign each sample to the nearest prototype
    distances = pairwise_distances(embeddings_np, prototype_embeddings_np, metric='euclidean')
    nearest_prototype_indices = np.argmin(distances, axis=1)
    
    # Compute silhouette score using the assigned clusters
    score = silhouette_score(embeddings_np, nearest_prototype_indices, metric='euclidean')
    
    return score

def elbow_method(embeddings, labels, model, N_range):
    """
    Computes the elbow method based on silhouette score and nearest medoid accuracy for values of N in a given range
    for all three methods ('global', 'per_class', 'per_class_variance').
    
    Args:
        embeddings (torch.Tensor): A tensor of shape (num_samples, embedding_dim) containing the image embeddings.
        labels (torch.Tensor): A tensor of shape (num_samples,) containing the labels corresponding to the embeddings.
        model (torch.nn.Module): A trained model that can be used for predictions.
        N_range (list): A list of values of N to evaluate.
        
    Returns:
        dict: A dictionary containing silhouette scores and accuracies for each value of N for each method.
    """
    methods = ['per_class', 'per_class_variance', 'global']
    all_results = {}
    
    for method in tqdm(methods):
        results = {'N': [], 'silhouette_score': [], 'accuracy': []}
        
        for N in N_range:
            # Find prototypes
            prototype_indices = find_prototypes(embeddings, labels, N, method=method)
            prototypes = embeddings[prototype_indices]
            prototype_labels = labels[prototype_indices]
            
            # Compute silhouette score for prototypes
            silhouette = compute_prototype_silhouette_score(embeddings, prototype_indices)
            
            # Compute nearest medoid accuracy
            accuracy = nearest_medoid_accuracy(embeddings, labels, prototypes, prototype_labels, model)
            
            # Store results
            results['N'].append(N)
            results['silhouette_score'].append(silhouette)
            results['accuracy'].append(accuracy)
        
        all_results[method] = results
    
    # Save all results to a dictionary
    with open('elbow_method_results.npy', 'wb') as f:
        np.save(f, all_results)
    
    # Plot the results for comparison
    plt.figure(figsize=(18, 12))
    for method in methods:
        plt.plot(all_results[method]['N'], all_results[method]['silhouette_score'], label=f'Silhouette Score ({method})', marker='o')
        plt.plot(all_results[method]['N'], all_results[method]['accuracy'], label=f'Accuracy ({method})', marker='o')
    
    plt.xlabel('Number of Prototypes (N)')
    plt.ylabel('Score')
    plt.title('Elbow Method for Prototypes Selection (Comparison of Methods)')
    plt.legend()
    plt.grid(True)
    plt.savefig('elbow_method_comparison.png')
    plt.show()
    
N_range=[20, 30, 40, 50, 60, 70, 80]
elbow_method(test_embeddings, test_labels, model, N_range)