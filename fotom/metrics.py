try:
    from fotom.embedding import Embedder
except:
    from embedding import Embedder
import torch
import numpy as np
import random
import networkx as nx
import os
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def WFD(graphs_1, graphs_2,
        embedder = None,
        fix_seeds = True, seed = 42):
    if fix_seeds:
        setup_seed(seed)
    if embedder is None:
        embedder = Embedder()
    if fix_seeds:
        setup_seed(seed)
    embeddings_1 = embedder(graphs_1)
    if fix_seeds:
        setup_seed(seed)
    embeddings_2 = embedder(graphs_2)

    distances = []
    for comp in range(embeddings_1.shape[1]):
        distances.append(wasserstein_distance(embeddings_1[:,comp], embeddings_2[:,comp]))
    return np.mean(distances)

if __name__ == "__main__":
    graphs_tree = []
    for _ in range(1000):
        graphs_tree.append(nx.random_tree(24))

    print(f"WFD two trees: {WFD(graphs_tree, graphs_tree)}")

    graphs_er = []
    for _ in range(1000):
        graphs_er.append(nx.erdos_renyi_graph(24, 0.1))

    print(f"WFD sparse ER/trees: {WFD(graphs_er, graphs_tree)}")

    graphs_er = []
    for _ in range(1000):
        graphs_er.append(nx.erdos_renyi_graph(24, 0.5))

    print(f"WFD dense ER/trees: {WFD(graphs_er, graphs_tree)}")


