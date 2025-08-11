import numpy as np
from sklearn.cluster import KMeans

def adjacency_matrix(edges, num_nodes):
    """Create an adjacency matrix from a list of edges."""
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=float)
    for edge in edges:
        u, v = edge
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1  # Assuming undirected graph
    return adj_matrix


def degree_matrix(adj_matrix):
    """Compute the degree matrix from an adjacency matrix."""
    degrees = np.sum(adj_matrix, axis=1)
    return np.diag(degrees)

def laplacian_matrix(adj_matrix, normalized=True):
    """Compute the Laplacian matrix from an adjacency matrix."""
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    laplacian = degree_matrix - adj_matrix
    
    if normalized:
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
        laplacian = d_inv_sqrt @ laplacian @ d_inv_sqrt
    
    return laplacian



def spectral_decomposition(laplacian):
    """Compute the eigenvalues and eigenvectors of the Laplacian matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    return eigenvalues, eigenvectors

