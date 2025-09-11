# Geometric-DL

A collection of **implementations**, **notebooks**, and **mathematical tools** for exploring **Geometric Deep Learning**.  
This repository includes step-by-step examples, reusable Python modules, and visualizations for understanding and applying techniques such as **Spectral Clustering**, **Graph Convolutional Networks (GCN)**, and other graph-based machine learning methods.

---

## 📂 Repository Structure (Incomplete)

```
In progress - the layout is just my thougth (8 Aug 2025)

Geometric-DL/
│
├── notebooks/              # Jupyter notebooks for experiments & tutorials
│   ├── spectral_clustering.ipynb
│   ├── gcn_basic.ipynb
│   └── ...
│
├── src/                    # Reusable Python modules 
│   ├── spectral/
│   │   ├── laplacian.py
│   │   ├── clustering.py
│   │   └── utils.py
│   ├── gcn/
│   │   ├── models.py
│   │   ├── train.py
│   │   └── utils.py
│   └── visualization/
│       ├── plot_graph.py
│       └── plot_clusters.py
│
├── data/                   # Example datasets
│   ├── karate_club.csv
│   └── ...
│
├── README.md
└── requirements.txt
```

---

## 🚀 Features

- **Spectral Methods**: Laplacian matrix computation, spectral clustering, and eigen-decomposition.
- **Graph Neural Networks**: Simple GCN layers, training scripts, and toy datasets.
- **Visualization Tools**: Graph plotting with `networkx` and clustering results with `matplotlib`.
- **Mathematical Utilities**: Helper functions for graph theory, linear algebra, and preprocessing.

---

## 📦 Installation

```bash
git clone https://github.com/ehsanrahnama/Geometric-DL.git
cd Geometric-DL
pip install -r requirements.txt
```

---

## 📝 Usage

Example: **Spectral Clustering on a small graph**

```python
from src.spectral.clustering import spectral_clustering
from src.spectral.laplacian import adjacency_matrix

edges = [(0, 1), (1, 2), (3, 4)]
A = adjacency_matrix(edges, num_nodes=5)
labels = spectral_clustering(A, k=2)
print(labels)
```

---

## 📚 Roadmap

- [x] Basic spectral clustering implementation
- [x] GCN from scratch
- [x] Graph attention networks (GAT)
- [ ] Real-world datasets (Cora, Citeseer)
- [ ] Advanced visualization layouts

---

## 🤝 Contributing

Contributions are welcome!  
Please open an issue or submit a pull request if you want to improve the repository.

---

## 📜 License

MIT License © 2025 Ehsan Rahnama
