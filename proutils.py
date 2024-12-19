import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import SpectralClustering
import kmedoids
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, silhouette_score
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from IF import *

def find_prototypes(embeddings, labels, N, method='global'):
    # Convert embeddings and labels to numpy arrays for use with scikit-learn
    embeddings_np = embeddings
    # .cpu().detach().numpy()
    labels_np = labels
    
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
            diss = euclidean_distances(class_embeddings)
            fp = kmedoids.fasterpam(diss, n_clusters, max_iter=100, random_state=42)
            
            # Get the medoid indices relative to the class embeddings
            medoid_indices = fp.medoids
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
            
            diss = euclidean_distances(class_embeddings)
            fp = kmedoids.fasterpam(diss, n_clusters, max_iter=100, random_state=42)
            
            # Get the medoid indices relative to the class embeddings
            medoid_indices = fp.medoids
            return medoid_indices
        
        # Parallelize the prototype finding for each class
        results = Parallel(n_jobs=-1)(delayed(find_class_prototypes)(cls) for cls in unique_classes)
        
        # Collect the results
        prototype_indices = np.hstack(results)
    
    elif method == 'global':
        # Find prototypes without considering class labels
        n_clusters = min(N, len(embeddings_np))
        diss = euclidean_distances(embeddings_np)
        fp = kmedoids.fasterpam(diss, n_clusters, max_iter=100, random_state=12)
        
        # Get the medoid indices relative to the class embeddings
        prototype_indices = fp.medoids
    
    else:
        raise ValueError("Invalid method specified. Choose from 'per_class', 'per_class_variance', or 'global'.")
    
    return prototype_indices

def nearest_medoid_accuracy(embeddings, labels, protos):
    
    prototype_embeddings=embeddings[protos]
    prototype_labels=labels[protos]
    # Convert embeddings to numpy array
    embeddings_np = embeddings
    # .cpu().detach().numpy()
    prototype_embeddings_np = prototype_embeddings
    # .cpu().detach().numpy()
    
    # Compute distances from each embedding to each prototype
    distances = cosine_similarity(embeddings_np, prototype_embeddings_np)
    nearest_medoid_indices = np.argmax(distances, axis=1)
    predicted_labels = prototype_labels[nearest_medoid_indices]
    
    # Compute accuracy of the nearest medoid classifier
    nearest_medoid_acc = accuracy_score(labels, predicted_labels)
    
    return nearest_medoid_acc



def compute_prototype_silhouette_score(embeddings, prototype_indices):

    # Convert embeddings to numpy arrays
    embeddings_np = embeddings 
    # .cpu().detach().numpy()
    prototype_embeddings_np = embeddings_np[prototype_indices]
    
    # Assign each sample to the nearest prototype
    distances = cosine_similarity(embeddings_np, prototype_embeddings_np)
    nearest_prototype_indices = np.argmax(distances, axis=1)
    
    # Compute silhouette score using the assigned clusters
    score = silhouette_score(embeddings_np, nearest_prototype_indices, metric='euclidean')
    
    return score

def elbow_method(embeddings, labels, N_range, X_test, mod_pred, proto=None, k=0):
    
    methods = ['global', 'per_class']
    all_results = {}
    
    for method in tqdm(methods):
        results = {'N': [], 'surrogate_fidelity': [], 'accuracy': []}
        # labels=labels.cpu().detach().numpy()
        for N in N_range:
            # Find prototypes
            if proto is None:
                prototype_indices = find_prototypes(embeddings, labels, N, method=method)                
            else:
                prototype_indices=list(proto[N].values())[k]
            prototypes = embeddings[prototype_indices]
            prototype_labels = labels[prototype_indices]
            
            # Compute silhouette score for prototypes
            surrogate = surrogate_fidelity(prototype_indices, X_test, mod_pred)
            
            # Compute nearest medoid accuracy
            accuracy = nearest_medoid_accuracy(embeddings, labels, prototypes, prototype_labels)
            
            # Store results
            results['N'].append(N)
            results['surrogate_fidelity'].append(surrogate)
            results['accuracy'].append(accuracy)
        
        all_results[method] = results
    # with open('elbow_method_results.npy', 'wb') as f:
    #     np.save(f, all_results)
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
    plt.show()
    




def custom_cosine_kernel(X, Y):
    # Calculate cosine similarity between each pair of rows in X and Y
    similarity_matrix = cosine_similarity(X,Y)
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            similarity_matrix[i, j] = similarity_matrix[i][j]
    return similarity_matrix


def aide(inf, train_embeddings, test_embeddings, iN, coverage=False):

    aide_emb = []
    for i, emb in enumerate(inf):
        # Sort indices based on embedding values
        scaler = MinMaxScaler()
        scaler.fit(emb.reshape(-1,1))
        emb=scaler.transform(emb.reshape(-1,1))
        ind = emb.reshape(-1).argsort()
        
        # Top 10 positive and negative indices
        top_pos_indices = ind[-2*iN:]
        # top_neg_indices = ind[:2*iN]

        # Compute cosine similarity for positive and negative indices
        pos_sim = cosine_similarity(train_embeddings[top_pos_indices], test_embeddings[i].reshape(1, -1)).flatten()
        # neg_sim = cosine_similarity(train_embeddings[top_neg_indices], test_embeddings[i].reshape(1, -1)).flatten()

        pos_idx= top_pos_indices[pos_sim.argsort()[-iN:]]
        # neg_idx= top_neg_indices[neg_sim.argsort()[-iN:]]
        if coverage:
            aide_emb.append(pos_idx)
        else:
            aide_emb.append(zip(pos_idx, emb.reshape(-1)[pos_idx]))
        # aide_emb.append(np.append(pos_idx, neg_idx))

    return aide_emb

def find_representative_samples(test_embeddings, train_embeddings, ifem, N, iN, alpha=None):
    aide_em=aide(ifem, train_embeddings, test_embeddings, iN)
    G = nx.Graph()
    for i, embs in enumerate(aide_em):
        G.add_node(i, bipartite=0)
        for ind, influence in embs:
            G.add_node(f'ex-{ind}', bipartite=1)
            G.add_edge(i, f'ex-{ind}', weight=influence)
    # Extract nodes and neighbors
    nodes_0 = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
    nodes_1 = [n for n, d in G.nodes(data=True) if d["bipartite"] == 1]

    if alpha:
        # Compute pairwise connection strengths
        num_nodes = len(nodes_0)
        connection_matrix = np.zeros((num_nodes, num_nodes))
        neighbors = {node: set(G.neighbors(node)) for node in nodes_0}
        for i, node1 in enumerate(nodes_0):
            for j, node2 in enumerate(nodes_0):
                if i != j:
                    # Compute shared neighbors count
                    common_neighbors=neighbors[node1] & neighbors[node2]
                    weits=sum([(G[i][k]['weight']+G[j][k]['weight'])/2 for k in common_neighbors])
                    connection_matrix[i, j] = weits/iN
        print("connection matrix is ready")
        # Compute semantic similarity
        embeddings_matrix = test_embeddings[nodes_0]
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Combine connection strength and semantic similarity (prioritizing connections)
        combined = np.vstack((connection_matrix, similarity_matrix))

        # Initialize and fit the scaler
        scaler = MinMaxScaler()
        scaler.fit(combined)

        # Transform each array
        scaled_array1 = scaler.transform(connection_matrix)
        scaled_array2 = scaler.transform(similarity_matrix)
        combined_matrix = (1-alpha)*scaled_array1 + alpha*scaled_array2
        
        # Perform clustering
        clustering = SpectralClustering(
            n_clusters=N,  # Adjust number of clusters as neededs
            affinity="precomputed",
            random_state=42
        )
        labels = clustering.fit_predict(combined_matrix)
        
        # Select representatives
        representatives = []
        for cluster_id in np.unique(labels):
            cluster_nodes = [nodes_0[i] for i in range(num_nodes) if labels[i] == cluster_id]
            # Select the node with the maximum connections
            representative = max(cluster_nodes, key=lambda n: sum(connection_matrix[nodes_0.index(n)]))
            representatives.append(representative)
        return representatives
    else:
        # neighbors = {node: set(G.neighbors(node)) for node in nodes_0}

        connection_matrix = np.zeros((len(nodes_0), len(nodes_1)))
        for i, test_node in enumerate(nodes_0):
            for j, train_node in enumerate(nodes_1):
                if G.has_edge(test_node, train_node):
                    connection_matrix[i, j] = G[test_node][train_node]['weight']    


        dissimilarity_matrix = squareform(pdist(connection_matrix, metric='cosine'))
        medoids=kmedoids.fasterpam(dissimilarity_matrix, N, max_iter=100, random_state=12)
        # Step 5: Extract representative nodes
        return medoids.medoids
        
def cluster_by_prototypes(data, prototypes):

    # Compute distances between each data point and each prototype
    sims = cosine_similarity(data, data[prototypes])
    
    # Assign each data point to the nearest prototype
    labels = np.argmax(sims, axis=1)
    
    return labels

def average_similarity(arrays):
    # Calculate cosine similarity for all pairs
    similarities = []
    num_arrays = len(arrays)
    similarity_matrix=cosine_similarity(arrays)

    for i in range(num_arrays):
        for j in range(i + 1, num_arrays):
            sim = similarity_matrix[i][j]
            similarities.append(sim)

    # Calculate the average similarity
    avg_similarity = np.mean(similarities) if similarities else 0
    return avg_similarity

def expected_inter_cluster_similarity(data, labels, distance_metric='cosine'):

    unique_labels = np.unique(labels)
    inter_cluster_similarities = []
    
    for label in unique_labels:
        cluster_points = data[labels==label]
        if len(cluster_points) > 1:
            inter_cluster_similarities.append(average_similarity(cluster_points))
    
    avg_inter_cluster_similarity = np.mean(inter_cluster_similarities)
    
    return avg_inter_cluster_similarity

