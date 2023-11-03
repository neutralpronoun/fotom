from fotom.embedding import Embedder
from fotom.model import load_encoder, load_transfer_model
import pytest
import networkx as nx
import numpy as np


def test_load_embedder():
    emb = Embedder()
    g = nx.random_tree(12)
    embedding = emb(g)
    assert isinstance(embedding, np.ndarray), "Resulting embedding is not numpy array from __call__ syntax"
    embedding = emb.transform(g)
    assert isinstance(embedding, np.ndarray), "Resulting embedding is not numpy array from .transform syntax"

def test_load_models():
    encoder_model = load_encoder()
    transfer_model = load_transfer_model()



