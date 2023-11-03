import numpy as np
import torch
import yaml
import os
from torch import Tensor
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool
from typing import Callable, Union
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_sparse import SparseTensor
import importlib.resources
import tempfile

def load_yaml_object(filename):
    package_name = 'fotom'  # Adjust to your package structure

    # Define the path to the file within your package
    file_path = filename

    try:
        # Use importlib.resources to load the file
        file_content = importlib.resources.read_text(package_name, file_path)

        # Parse the YAML content
        loaded_data = yaml.safe_load(file_content)

        return loaded_data
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {e}")


def load_torch_object(filename):
    package_name = 'fotom'  # Adjust to your package structure

    # Define the path to the file within your package
    file_path = filename

    try:
        # Use importlib.resources to load the file
        file_content = importlib.resources.read_binary(package_name, file_path)

        # Create a temporary file to save the content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_content)

        # Use torch.load to load the content from the temporary file
        loaded_data = torch.load(temp_file.name, map_location=torch.device('cpu'))

        # Clean up the temporary file
        os.remove(temp_file.name)

        return loaded_data
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {e}")

def initialize_edge_weight(data):
	data.edge_weight = torch.ones(data.edge_index.shape[1], dtype=torch.float)
	return data

def load_encoder():
    checkpoint = "fotom.pt"
    cfg_name = checkpoint.split('.')[0] + ".yaml"

    args = load_yaml_object(cfg_name)

    model = Encoder(emb_dim=args["emb_dim"]["value"], num_gc_layers=args["num_gc_layers"]["value"], drop_ratio=args["drop_ratio"]["value"],
                    pooling_type=args["pooling_type"]["value"])

    model_dict = load_torch_object("fotom.pt")
    model.load_state_dict(model_dict['encoder_state_dict'], strict=False)

    return model

def load_transfer_model(linear_layers = (300,), output_dim = 1):
    checkpoint = "fotom.pt"
    cfg_name = checkpoint.split('.')[0] + ".yaml"
    args = load_yaml_object(cfg_name)

    model = TransferModel(Encoder(emb_dim=args["emb_dim"]["value"], num_gc_layers=args["num_gc_layers"]["value"], drop_ratio=args["drop_ratio"]["value"],
                                  pooling_type=args["pooling_type"]["value"]),
                          linear_layers=linear_layers, output_dim=output_dim)

    model_dict = load_torch_object("fotom.pt")
    model.load_state_dict(model_dict['encoder_state_dict'], strict=False)

    return model

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class GINEConv(MessagePassing):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINEConv, self).__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor, edge_weight) -> Tensor:
        return F.relu(x_j + edge_attr) if edge_weight is None else F.relu(x_j + edge_attr) * edge_weight.view(-1, 1)



    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class Encoder(torch.nn.Module):
	def __init__(self, emb_dim=300, num_gc_layers=5, drop_ratio=0.0, pooling_type="standard", is_infograph=False):
		super(Encoder, self).__init__()

		self.pooling_type = pooling_type
		self.emb_dim = emb_dim
		self.num_gc_layers = num_gc_layers
		self.drop_ratio = drop_ratio
		self.is_infograph = is_infograph

		self.out_node_dim = self.emb_dim
		if self.pooling_type == "standard":
			self.out_graph_dim = self.emb_dim
		elif self.pooling_type == "layerwise":
			self.out_graph_dim = self.emb_dim * self.num_gc_layers
		else:
			raise NotImplementedError

		self.convs = torch.nn.ModuleList()
		self.bns = torch.nn.ModuleList()

		self.atom_encoder = AtomEncoder(emb_dim)
		self.bond_encoder = BondEncoder(emb_dim)

		for i in range(num_gc_layers):
			nn = Sequential(Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), ReLU(), Linear(2*emb_dim, emb_dim))
			conv = GINEConv(nn)
			bn = torch.nn.BatchNorm1d(emb_dim)
			self.convs.append(conv)
			self.bns.append(bn)

		self.init_emb()

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):
		x = self.atom_encoder(x.to(torch.int))
		edge_attr = self.bond_encoder(edge_attr.to(torch.int))
		# compute node embeddings using GNN
		xs = []
		for i in range(self.num_gc_layers):
			x = self.convs[i](x, edge_index, edge_attr, edge_weight)
			x = self.bns[i](x)
			if i == self.num_gc_layers - 1:
				# remove relu for the last layer
				x = F.dropout(x, self.drop_ratio, training=self.training)
			else:
				x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
			xs.append(x)

		# compute graph embedding using pooling
		if self.pooling_type == "standard":
			xpool = global_add_pool(x, batch)
			return xpool, x

		elif self.pooling_type == "layerwise":
			xpool = [global_add_pool(x, batch) for x in xs]
			xpool = torch.cat(xpool, 1)
			if self.is_infograph:
				return xpool, torch.cat(xs, 1)
			else:
				return xpool, x
		else:
			raise NotImplementedError

	def get_embeddings(self, loader, device, is_rand_label=False):
		ret = []
		y = []
		with torch.no_grad():
			for i, data in enumerate(loader):

				if isinstance(data, list):
					data = data[0].to(device)
				# elif isinstance(data, tuple):
				# 	data = data[1].to(device)

				data = data.to(device)
				batch, x, edge_index, edge_attr = data.batch, data.x, data.edge_index, data.edge_attr

				# print(edge_index.max(), x.shape)

				edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

				if x is None:
					x = torch.ones((batch.shape[0], 1)).to(device)
				x, _ = self.forward(batch, x, edge_index, edge_attr, edge_weight)

				ret.append(x.cpu().numpy())
				# print(x.shape)
				# print(data.y, data.y.shape)
				try:
					if is_rand_label:
						y.append(data.rand_label.cpu().numpy())
					else:
						y.append(data.y.cpu().numpy())
				except AttributeError:
					y.append(torch.ones(x.shape[0]).to(torch.float))
		ret = np.concatenate(ret, 0)
		y = np.concatenate(y, 0)
		return ret, y

class TransferModel(torch.nn.Module):
    def __init__(self, encoder, linear_layers=(300,), output_dim=300):
        super(TransferModel, self).__init__()

        self.encoder = encoder
        self.input_proj_dim = self.encoder.out_graph_dim

        output_layers = [Linear(self.input_proj_dim, linear_layers[0]), ReLU(inplace=True)]
        if len(linear_layers) > 1:
            for i_layer, layer_dim in enumerate(linear_layers[:-1]):
                output_layers.append(Linear(linear_layers[i_layer], linear_layers[i_layer+1]))
                output_layers.append(ReLU(inplace=True))
        output_layers += [Linear(linear_layers[-1], output_dim)]

        self.output_layer = Sequential(*output_layers)

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):
        z, node_emb = self.encoder(batch, x, edge_index, edge_attr, edge_weight)

        z = self.output_layer(z)
        # z shape -> Batch x proj_hidden_dim
        return z, node_emb


if __name__ == "__main__":
    model = load_encoder()
    print(model)
    model = load_transfer_model(linear_layers=(300, 100, 25, 65,))
    print(model)
