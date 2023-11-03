# FoToM

A pip package for FoToM (**Fo**undational **To**pology **M**odels), pretrained models for graph deep learning.

![](https://raw.githubusercontent.com/neutralpronoun/fotom/main/fotom/assets/embedding-github.png)

FoToM is a model pre-trained with contrastive learning on node-label-free graphs, but we show that under fine-tuning on node-labelled data there is still a significant performance increase.

### Data Formats

 - Data can be passed to Embedder as `networkx` graphs or `pytorch_geometric.data.Data` objects
 - `pytorch_geometric.loader.DataLoader` loaders are also compatible for more advanced users
 - `networkx` graphs can have node or edge attributes, but only of `dim=1`, ie labels
 - If no node or edge labels are passed they are filled with `torch.ones(data.num_nodes)` or `torch.ones(data.num_edges)` tensors

### Embeddings

The simplest use case is to use our pre-trained model to produce vector graph embeddings.
In code this can be done through:

```
import networkx as nx
from fotom.embedding import Embedder

# Get some dummy graph data
graphs = []
for _ in range(1000):
    graphs.append(nx.random_tree(12))

# Instatiate Embedder, this loads our pre-trained checkpoint
embedder = Embedder(device = "cpu", encoder = None)

# Using __call__ syntax:
embedding = embedder(graphs)

# Using .transform:
embedding = embedder.transform(graphs)
```

The returned embeddings are of shape `(N_samples, output_dim)`.

The embedder only has two init arguments:
 - `device`: the device to use, probably "gpu" or "cpu", default "cpu"
 - 'encoder': a different encoder - see below



### Models

A model with (untrained) linear output layers, for transfer learning, is available through:

```
from fotom.model import load_transfer_model

model = load_transfer_model(linear_layers=(300,), output_dim=1)
```

The model here, loaded with pre-trained weights through `load_transfer_model`, has two keyword arguments:
 - `linear_layers`: a tuple of the number of neurons in each layer following the encoder
 - `output_dim`: the number of neurons in the final output layer

### Advanced Usage

FoToM essentially comes down to two classes:

`fotom.model.Encoder`, the encoder without pre-trained weights, taking arguments:
- `emb_dim=300`: the dimension of the encodings
- `num_gc_layers=5`: the number of layers in the encoder architecture
- `drop_ratio=0.0`: the drop ratio during training
- `pooling_type="standard"`: 'standard' or 'layerwise', how pooling happens between nodes before output

These parameters are set through a config file when using `load_encoder`

`fotom.model.TransferModel`, a wrapper for `fotom.model.Encoder` with additional linear layers after the encoder.
This takes the following arguments:
- `encoder`: (probably) an `fotom.model.Encoder` instance
- `linear_layers=(300,)`: the dimensions of each (non-output) linear layer following the encoder
- `output_dim=300`: the dimensions of the final output

## Credits


FoToM is a work by Alex Davies, Riku Green, Nirav Ajmeri and Telmo de Menezes e Silva Filho, all academics at the University of Bristol.

FoToM is an extension of the AD-GCL work by Suresh et. al., <https://arxiv.org/abs/2106.05819>.

If you use FoToM in your work, please give due credit to both sets of authors in any resulting publication.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
