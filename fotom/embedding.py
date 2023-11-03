import torch

from fotom.model import load_encoder, load_transfer_model, TransferModel, Encoder
import networkx as nx
from tqdm import tqdm
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader

class Embedder:
    """
    Produces embeddings for
    """
    def __init__(self, device = "cpu", encoder = None):
        if encoder is None:
            self.encoder = load_encoder()
        elif type(encoder) == Encoder:
            print("Using passed encoder")
            self.encoder = encoder
        elif type(encoder) == TransferModel:
            print("Using encoder from passed transfer model")
            self.encoder = encoder.encoder

        self.device = device
        self.encoder.to(device)

    def __call__(self, data):
        return self.transform(data)

    def convert_from_networkx(self, data):
        pbar = tqdm(data, desc = "Converting from networkx", leave=False)
        for i_graph, graph in enumerate(pbar):
            try:
                data[i_graph] = from_networkx(graph,
                                              group_node_attrs = all,
                                              group_edge_attrs = all)
            except:
                data[i_graph] = from_networkx(graph)
        return data

    def check_data(self, data):
        for i_graph, graph in enumerate(data):
            if graph.x is None:
                x = torch.ones(graph.num_nodes).reshape(-1,1)
                new_graph = graph
                new_graph.x = x
                data[i_graph] = new_graph
            if graph.edge_attr is None:
                E = torch.ones(graph.num_edges).reshape(-1,1)
                new_graph = graph
                new_graph.edge_attr = E
                data[i_graph] = new_graph
        return data

    def transform(self, data, batch_size = 64):
        if type(data) != list:
            data = self.convert_from_networkx([data])
        elif type(data[0]) == nx.Graph:
            data = self.convert_from_networkx(data)

        if type(data) != DataLoader:
            data = self.check_data(data)
            data = DataLoader(data, batch_size=batch_size)
        embeddings = self.encoder.get_embeddings(data, device=self.device)[0]

        return embeddings

if __name__ == "__main__":
    # embedder = Embedder()
    graphs = []
    for _ in range(1000):
        graphs.append(nx.random_tree(12))

    # model = load_transfer_model()
    embedder = Embedder()
    print(embedder(graphs).shape)
