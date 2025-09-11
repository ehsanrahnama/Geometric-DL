import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch_geometric.datasets import KarateClub
import networkx as nx
import matplotlib.pyplot as plt
from model import DeepGCN  # Import the DeepGCN model





# ---------- Utility: Create Normalized Adjacency ----------
def normalize_adjacency(edge_index, num_nodes):
    # Build adjacency matrix
    A = torch.zeros((num_nodes, num_nodes))
    for i, j in edge_index.T:
        A[i, j] = 1
    # Add self-loops
    A = A + torch.eye(num_nodes)
    # Compute D^{-1/2} A D^{-1/2}
    D = torch.diag(torch.pow(A.sum(1), -0.5))
    A_hat = D @ A @ D
    return A_hat


# ---------- Training Function ----------
def train(model, X, A_hat, y, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X, A_hat)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            pred = out.argmax(dim=1)
            acc = (pred == y).float().mean().item()
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")

    return model


# ---------- Main ----------
if __name__ == "__main__":
    # Load dataset
    dataset = KarateClub()
    data = dataset[0]

    # Features & Labels
    X = data.x
    y = data.y

    # Normalized adjacency
    A_hat = normalize_adjacency(data.edge_index, data.num_nodes)

    # Model: 3 hidden layers
    model = DeepGCN(
        in_feats=X.shape[1],
        hidden_sizes=[32, 16, 8],
        out_feats=dataset.num_classes
    )

    # Train
    trained_model = train(model, X, A_hat, y)

    # Evaluate
    trained_model.eval()
    logits = trained_model(X, A_hat)
    pred = logits.argmax(dim=1)

    acc = (pred == y).float().mean().item()
    print(f"\nFinal Accuracy: {acc:.4f}")

    # Optional: visualize result
    G = nx.Graph()
    G.add_edges_from(data.edge_index.T.tolist())
    plt.figure(figsize=(6, 5))
    nx.draw_networkx(G, node_color=pred.numpy(), cmap=plt.cm.Set1, with_labels=False)
    plt.title("DeepGCN Predictions")
    plt.show()
