
import numpy as np
from sklearn.cluster import KMeans
from .laplacian import adjacency_matrix, laplacian_matrix, spectral_decomposition

def spectral_clustering(edges, num_nodes, k=2):
    """Perform spectral clustering on a graph defined by edges."""
    adj_matrix = adjacency_matrix(edges, num_nodes)
    laplacian = laplacian_matrix(adj_matrix)
    eigenvalues, eigenvectors = spectral_decomposition(laplacian)

    # Select the first k eigenvectors (excluding the first one which is trivial)
    selected_vectors = eigenvectors[:, 1:k+1]
    
    # Perform k-means clustering on the selected eigenvectors
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(selected_vectors)

    return labels, eigenvalues, selected_vectors