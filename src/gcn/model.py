import torch
import torch.nn as nn
import torch.nn.functional as F


# ‌Basic Layer of GCN
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, X, A_hat):
        out = torch.mm(A_hat, X)   # Messaging (propagation)
        out = self.linear(out)     # Linear transformation
        return out


# GCN Model with Multiple Layers
class DeepGCN(nn.Module):
    def __init__(self, in_feats, hidden_sizes, out_feats):
        super(DeepGCN, self).__init__()

        layers = []
        input_dim = in_feats
        for hidden_dim in hidden_sizes:
            layers.append(GCNLayer(input_dim, hidden_dim))
            input_dim = hidden_dim
        # Output layer    
        layers.append(GCNLayer(input_dim, out_feats))

        self.layers = nn.ModuleList(layers)

    def forward(self, X, A_hat):
        h = X
        for i, layer in enumerate(self.layers):
            h = layer(h, A_hat)
            if i < len(self.layers) - 1:   # Exclude activation after last layer
                h = F.relu(h)
        return h



if __name__ == "__main__":
    # Example usage
    # Assuming A is the adjacency matrix and X is the feature matrix
    A = torch.tensor([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]], dtype=torch.float32)

    X = torch.tensor([[1, 0],
                        [0, 1],
                        [1, 1]], dtype=torch.float32)
    # Adding self-loops to adjacency matrix
    I = torch.eye(A.size(0))
    A_hat = A + I
    # Normalizing A_hat
    D_hat = torch.diag(torch.sum(A_hat, dim=1))
    D_hat_inv_sqrt = torch.pow(D_hat, -0.5)
    D_hat_inv_sqrt[torch.isinf(D_hat_inv_sqrt)] = 0.
    A_hat = torch.mm(torch.mm(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)    

    model = DeepGCN(
        in_feats=A.shape[0], 
        hidden_sizes=[32, 16, 8], 
        out_feats=2
    )
