from sklearn import metrics
def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = dgl.node_subgraph(g, g.ndata['train_mask'])
    val_g = dgl.node_subgraph(g, g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g
def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def calculate_metrics(labels, predictions):
    assert labels.shape == predictions.shape
    prec,rec,f1,num = metrics.precision_recall_fscore_support(labels, predictions, average='binary', pos_label=1)
    auc = metrics.roc_auc_score(labels, predictions, average=None)
#     results = {'Precision':prec, 'Recall':rec, 'F1':f1, 'AUC':auc}
    return (prec,rec,f1,auc)

def evaluate(model, g, nfeat, labels, val_nid, device='cpu'):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
        if pred.shape[-1] != 2:
            pred = pred.max(1)[1]
    return calculate_metrics(labels[val_nid].to(pred.device), pred[val_nid])
def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    
    return batch_inputs, batch_labels


def evaluate_light(model, g, loader_val, loader_test, sens_selector, in_sentences, out_sentences, device='cpu'):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        preds = []
        labels = []
        for loader_id, (sub_graph, subset, batch_size) in enumerate(loader_val):
            sub_graph = sub_graph.to(device)
            in_pack, lens_in = sens_selector.select(subset,in_sentences, g.lens_in)
            out_pack, lens_out = sens_selector.select(subset,  out_sentences, g.lens_out)
            in_pack = in_pack.to(device)
            out_pack = out_pack.to(device)
            batch_pred, *_ = model(in_pack, out_pack, lens_in, lens_out, sub_graph )
            preds.append(batch_pred.cpu()[:batch_size])
            labels.append(g.labels.cpu()[subset][:batch_size])
        preds_val = torch.cat(preds, 0)
        labels_val = torch.cat(labels, 0)
        preds_val = preds_val.max(1)[1]
        results_val = calculate_metrics(labels_val, preds_val)
        results_val = results = {'Precision_val':results_val[0], 
                                 'Recall_val':results_val[1], 
                                 'F1_val':results_val[2], 
                                 'AUC_val':results_val[3]}
        preds = []
        labels = []
        for loader_id, (sub_graph, subset, batch_size) in enumerate(loader_test):
            sub_graph = sub_graph.to(device)
            in_pack, lens_in = sens_selector.select(subset,in_sentences, g.lens_in)
            out_pack, lens_out = sens_selector.select(subset,  out_sentences, g.lens_out)
            in_pack = in_pack.to(device)
            out_pack = out_pack.to(device)
            batch_pred, *_ = model(in_pack, out_pack, lens_in, lens_out, sub_graph )
            preds.append(batch_pred.cpu()[:batch_size])
            labels.append(g.labels.cpu()[subset][:batch_size])
        preds_test = torch.cat(preds, 0)
        labels_test = torch.cat(labels, 0)
        preds_test = preds_test.max(1)[1]
        torch.save(preds_test, './data/preds_test.pt')
        mlflow.log_artifact('./data/preds_test.pt', artifact_path='results') 
        results_test = calculate_metrics(labels_test, preds_test)
        results_test = results = {'Precision_test':results_test[0], 
                         'Recall_test':results_test[1], 
                         'F1_test':results_test[2], 
                         'AUC_test':results_test[3]}
        return results_val, results_test
    
    
from typing import List, Optional, Tuple, NamedTuple, Union, Callable

import torch
from torch import Tensor
from torch_sparse import SparseTensor


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)


class DualNeighborSampler(torch.utils.data.DataLoader):
    r"""The neighbor sampler from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, which allows
    for mini-batch training of GNNs on large-scale graphs where full-batch
    training is not feasible.

    Given a GNN with :math:`L` layers and a specific mini-batch of nodes
    :obj:`node_idx` for which we want to compute embeddings, this module
    iteratively samples neighbors and constructs bipartite graphs that simulate
    the actual computation flow of GNNs.

    More specifically, :obj:`sizes` denotes how much neighbors we want to
    sample for each node in each layer.
    This module then takes in these :obj:`sizes` and iteratively samples
    :obj:`sizes[l]` for each node involved in layer :obj:`l`.
    In the next layer, sampling is repeated for the union of nodes that were
    already encountered.
    The actual computation graphs are then returned in reverse-mode, meaning
    that we pass messages from a larger set of nodes to a smaller one, until we
    reach the nodes for which we originally wanted to compute embeddings.

    Hence, an item returned by :class:`NeighborSampler` holds the current
    :obj:`batch_size`, the IDs :obj:`n_id` of all nodes involved in the
    computation, and a list of bipartite graph objects via the tuple
    :obj:`(edge_index, e_id, size)`, where :obj:`edge_index` represents the
    bipartite edges between source and target nodes, :obj:`e_id` denotes the
    IDs of original edges in the full graph, and :obj:`size` holds the shape
    of the bipartite graph.
    For each bipartite graph, target nodes are also included at the beginning
    of the list of source nodes so that one can easily apply skip-connections
    or add self-loops.

    .. warning::

        :class:`~torch_geometric.loader.NeighborSampler` is deprecated and will
        be removed in a future release.
        Use :class:`torch_geometric.loader.NeighborLoader` instead.

    .. note::

        For an example of using :obj:`NeighborSampler`, see
        `examples/reddit.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        reddit.py>`_ or
        `examples/ogbn_products_sage.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        ogbn_products_sage.py>`_.

    Args:
        edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
            :obj:`torch_sparse.SparseTensor` that defines the underlying graph
            connectivity/message passing flow.
            :obj:`edge_index` holds the indices of a (sparse) symmetric
            adjacency matrix.
            If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its shape
            must be defined as :obj:`[2, num_edges]`, where messages from nodes
            :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
            (in case :obj:`flow="source_to_target"`).
            If :obj:`edge_index` is of type :obj:`torch_sparse.SparseTensor`,
            its sparse indices :obj:`(row, col)` should relate to
            :obj:`row = edge_index[1]` and :obj:`col = edge_index[0]`.
            The major difference between both formats is that we need to input
            the *transposed* sparse adjacency matrix.
        sizes ([int]): The number of neighbors to sample for each node in each
            layer. If set to :obj:`sizes[l] = -1`, all neighbors are included
            in layer :obj:`l`.
        node_idx (LongTensor, optional): The nodes that should be considered
            for creating mini-batches. If set to :obj:`None`, all nodes will be
            considered.
        num_nodes (int, optional): The number of nodes in the graph.
            (default: :obj:`None`)
        return_e_id (bool, optional): If set to :obj:`False`, will not return
            original edge indices of sampled edges. This is only useful in case
            when operating on graphs without edge features to save memory.
            (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(self, edge_index: Union[Tensor, SparseTensor],
                 sizes: List[int], node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, return_e_id: bool = True,
                 transform: Callable = None, **kwargs):

        edge_index = edge_index.to('cpu')

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if 'dataset' in kwargs:
            del kwargs['dataset']

        # Save for Pytorch Lightning...
        src, dst = edge_index
        edge_index_inverse = torch.vstack((dst, src))
        edge_index = torch.vstack((src, dst))
        edge_index_non_inverse = torch.cat((edge_index, edge_index), -1)
        edge_index_inverse = torch.cat((edge_index, edge_index_inverse), -1)
        self.edge_index = edge_index_non_inverse
        self.node_idx = node_idx
        self.num_nodes = num_nodes

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None

        # Obtain a *transposed* `SparseTensor` instance.
        if not self.is_sparse_tensor:
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.bool):
                num_nodes = node_idx.size(0)
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.long):
                num_nodes = max(int(edge_index.max()), int(node_idx.max())) + 1
            if num_nodes is None:
                num_nodes = int(edge_index.max()) + 1
            self.num_nodes = num_nodes
            value = torch.arange(edge_index_inverse.size(1)) if return_e_id else None
            self.adj_t = SparseTensor(row=edge_index_inverse[0], col=edge_index_inverse[1],
                                      value=value,
                                      sparse_sizes=(num_nodes, num_nodes)).t()
        else:
            adj_t = edge_index_inverse
            if return_e_id:
                self.__val__ = adj_t.storage.value()
                value = torch.arange(adj_t.nnz())
                adj_t = adj_t.set_value(value, layout='coo')
            self.adj_t = adj_t

        self.adj_t.storage.rowptr()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super().__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            if self.__val__ is not None:
                adj_t.set_value_(self.__val__[e_id], layout='coo')

            if self.is_sparse_tensor:
                adjs.append(Adj(adj_t, e_id, size))
            else:
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([col, row], dim=0)
                adjs.append(EdgeIndex(edge_index, e_id, size))

        adjs = adjs if len(adjs) == 1 else adjs[::-1]
        eids = adjs[0].e_id
        edge_index = self.edge_index[:, eids]
        node_idx = torch.zeros(self.num_nodes, dtype=torch.long)
        node_idx[n_id] = torch.arange(n_id.size(0))
        edge_index = node_idx[edge_index]
        return edge_index, n_id, batch_size

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sizes={self.sizes})'
    
    
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence
from torch_geometric.utils import subgraph, contains_self_loops, to_undirected,  remove_self_loops
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class PreSentences_light(Dataset):
    def __init__(self, train = False, train_mask = None):
        if train:
            assert train_mask is not None
        self.train = train
        self.train_mask = train_mask
    def select(self,idx, sens, lens):
#         if not self.train:
#             lens_selected = lens[idx]
#             sens_selected = sens[idx]
#         else:
        lens_selected = lens[idx]
        sens_selected = sens[idx]
#         res = pad_sequence(sens_selected, batch_first=True)
#         packed_res = pack_padded_sequence(res, lens_selected, batch_first=True, enforce_sorted=False)
        return sens_selected, lens_selected
    
    
    
    
    
from typing import Union, List, Dict, Tuple, Callable, Optional
from torch_geometric.typing import NodeType, EdgeType
from torch.utils.data import DataLoader
import torch
from torch import Tensor
from tqdm import tqdm
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.utils import edge_type_to_str
from torch_geometric.loader.utils import to_csc, to_hetero_csc
from torch_geometric.loader.utils import filter_data

   
    
class NeighborLoader_light(torch.utils.data.DataLoader):

    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        input_nodes: Union[Optional[Tensor], NodeType,
                           Tuple[NodeType, Optional[Tensor]]] = None,
        replace: bool = False,
        directed: bool = True,
        transform: Callable = None,
        **kwargs,
    ):
        if kwargs.get('num_workers', 0) > 0:
            torch.multiprocessing.set_sharing_strategy('file_system')
            kwargs['persistent_workers'] = True

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if 'dataset' in kwargs:
            del kwargs['dataset']

        self.data = data
        self.num_neighbors = num_neighbors
        self.input_nodes = input_nodes
        self.replace = replace
        self.directed = directed
        self.transform = transform

        if isinstance(data, Data):
            self.sample_fn = torch.ops.torch_sparse.neighbor_sample
            # Convert the graph data into a suitable format for sampling.
            self.colptr, self.row, self.perm = to_csc(data)
            assert isinstance(num_neighbors, (list, tuple))
            assert input_nodes is None or isinstance(input_nodes, Tensor)
            if input_nodes is None:
                self.input_nodes = torch.arange(data.num_nodes)
            elif input_nodes.dtype == torch.bool:
                self.input_nodes = input_nodes.nonzero(as_tuple=False).view(-1)
            
            self.sample = DualNeighborSampler(data, num_neighbors,
                                                    replace, directed,
                                                    input_node_type)  
            
            
            
            
            super().__init__(self.input_nodes.tolist(), collate_fn=self.sample,
                             **kwargs)



    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    
    
    
    
import torch
from torch import Tensor

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

class BalancedSampler(torch.utils.data.Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, y, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.pos_index = (y==1).nonzero().reshape(-1)
        self.neg_index = (y==0).nonzero().reshape(-1)
        self.n_pos = len(self.pos_index)
#         self.data_source = data_source
        self.replacement = replacement
        self._num_samples = 2*len(self.pos_index)
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return 2*self.n_pos
        return 2*self.n_pos

    def __iter__(self) -> Iterator[int]:
        n = 2*self.n_pos
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
            
            neg_seed =  int(torch.empty((), dtype=torch.int64).random_().item())
            neg_generator = torch.Generator()
            neg_generator.manual_seed(neg_seed)
            chosen_neg = self.neg_index[torch.randperm(len(self.neg_index), generator=neg_generator)[:self.n_pos]]
            data_source = torch.cat((self.pos_index, chosen_neg), -1)
        else:
            generator = self.generator
#         print(len(data_source))

        for _ in range(self.num_samples // n): # only iterate once  self.num_samples==n
            yield from data_source[torch.randperm(n, generator=generator)].tolist()
        yield from data_source[torch.randperm(n, generator=generator)].tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples