import networkx as nx
import numpy as np

def wandb_cfg_to_actual_cfg(wandb_cfg):
    """
    Retrive wandb config from saved file
    Args:
        original_cfg: the config from this run
        wandb_cfg: the saved config from the training run

    Returns:
        a config with values updated to those from the saved training run
    """
    # original_keys = list(vars(original_cfg).keys())
    original_cfg = {}
    wandb_keys = list(wandb_cfg.keys())
    print(wandb_cfg)

    for key in wandb_keys:
        if len(wandb_cfg[key]) <= 1:
            continue
        vars(original_cfg)[key] = wandb_cfg[key]['value']

    return original_cfg

def better_to_nx(data):
    """
    Converts a pytorch_geometric.data.Data object to a networkx graph,
    robust to nodes with no edges, unlike the original pytorch_geometric version

    Args:
        data: pytorch_geometric.data.Data object

    Returns:
        g: a networkx.Graph graph
        labels: torch.Tensor of node labels
    """
    edges = data.edge_index.T.cpu().numpy()
    labels = data.x[:,0].cpu().numpy()

    g = nx.Graph()
    g.add_edges_from(edges)

    for ilabel in range(labels.shape[0]):
        if ilabel not in np.unique(edges):
            g.add_node(ilabel)

    return g, labels
