import pickle 
import json
import os


dir='/data/zhihao/Bitcoin'
data_dir='/data/zhihao/Bitcoin'
def load_pickle(fname):
    with open(os.path.join(data_dir,fname), 'rb') as f:
        return pickle.load(f)
def save_obj(obj, name ):
    with open( os.path.join(data_dir,name)+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def save_json(data, name):
    with open( os.path.join(data_dir,name)+'.json', 'w') as f:
        json.dump(data,f)
def load_json(fname):
    with open( os.path.join(data_dir,fname), 'r') as f:
        return json.load(f)
    
    
from torch_geometric.utils import subgraph
from torch_geometric.transforms import RandomNodeSplit
from torch import Tensor
import torch_geometric
from torch_geometric.data import Data
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
import torch_geometric

def subdata(data: torch_geometric.data.data.Data, subset, subedges=None, relabel_nodes=True):
    device = data.edge_index.device
    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1]
    if isinstance(subset, (list, tuple)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        node_mask = subset
        num_nodes = node_mask.size(0)

        if relabel_nodes:
            node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
                                   device=device)
            node_idx[subset] = torch.arange(subset.sum().item(), device=device)
    else:
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        node_mask[subset] = 1

        if relabel_nodes:
            node_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
            node_idx[subset] = torch.arange(subset.size(0), device=device)
            
    sub_data = Data()
    
    if subedges is not None:
        if subedges.dtype == torch.bool:
            assert subedges.shape[0] == num_edges
        else:
            assert subedges.max() < num_edges
            
    # Get subgraph nodes and edges feature
    for key, item in data:
        if key in ['num_nodes', 'edge_index']:
            continue
        if isinstance(item, Tensor) and item.size(0) == num_nodes:
            sub_data[key] = item[subset]
        elif isinstance(item, Tensor) and item.size(0) == num_edges:
            if subedges is None:
                edge_index, sub_data[key] = subgraph(subset, data.edge_index, data[key], relabel_nodes=relabel_nodes)  
            else:
                sub_data[key] = item[subedges]
        else:
            sub_data[key] = item
    if subedges is None:
        sub_data.edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=relabel_nodes)   
    else:
        edge_index = data.edge_index[:, subedges]
        if relabel_nodes:
            edge_index = node_idx[edge_index]
        sub_data.edge_index = edge_index
    return sub_data

def load_pseudo_pg(datadir_path='./data/2015', use_unlabeled = 'SEMI', scale='minmax', graph_type = 'MultiDi', feature_type ='edge', train_rate=0.5, anomaly_rate=None, random_state=5211):
    # fix random seeds
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
    data = torch.load(os.path.join(datadir_path, 'MultiGraph_2015_3rd.pt'))
    if anomaly_rate:
        n_neg = (data.y == 0).sum().item()
        pos_ids = (data.y == 1).nonzero().view(-1).numpy()
        np.random.shuffle(pos_ids)
        drop_pos_ids = pos_ids[int(n_neg*anomaly_rate/(1-anomaly_rate)):]
        data.y[drop_pos_ids] = -1
#     X = data.X
    labels = data.y# label is here
    n_nodes = len(labels)
    all_id = np.arange(n_nodes)
    # label_mask is used in semi-supervised setting to identify which nodes are labeled ones (attend loss 
    # calculation) while others are unlabeled ones 
    if use_unlabeled == 'ALL': # regard unlabeled as normal users
        labels = np.where(labels == -1, 0, labels)
        label_mask = torch.ones(len(labels)).bool()
    elif use_unlabeled == 'NONE': 
        labels += 1 
        nodes_id = labels.nonzero().reshape(-1)
        labels = labels[nodes_id]
        labels -= 1
        X = X[nodes_id]
        data = subdata(data, nodes_id,relabel_nodes=True)
        label_mask = torch.ones(len(labels)).bool()
    elif use_unlabeled == 'SEMI': 
        labels += 1 
        label_id = labels.nonzero().reshape(-1)
        label_mask = torch.zeros(n_nodes )
        label_mask[label_id ] = 1
        label_mask  = label_mask.bool()
        labels -= 1
        
    n_nodes = len(labels) # refresh n_nodes
    all_id = np.arange(n_nodes)
    
#     if scale == 'norm':
#         X_norm = preprocessing.normalize(X, norm='l1', axis=0)
# #         data.edge_attr = preprocessing.normalize(data.edge_attr, norm='l1', axis=0)
#     elif scale == 'std':
#         scaler = preprocessing.StandardScaler()
#         X_norm = scaler.fit_transform(X)
#         scaler = preprocessing.StandardScaler()
#         data.edge_attr = torch.tensor(scaler.fit_transform(data.edge_attr)).float()
#     elif scale == 'minmax':
#         scaler = preprocessing.MinMaxScaler()
#         X_norm = scaler.fit_transform(X)
#         scaler = preprocessing.MinMaxScaler()
#         data.edge_attr = torch.tensor(scaler.fit_transform(data.edge_attr)).float()
#     elif scale == 'none':
#         X_norm = X
#     data['features'] = torch.tensor(X_norm).float()
    data['labels'] = torch.tensor(labels, dtype = int)

    # Split data into train/val/test
    
    # data.edge_index, data.edge_attr = add_self_loops(edge_index=data.edge_index, edge_attr =data.edge_attr, fill_value='mean')
    # # Split data into train/val/test
    np.random.shuffle(all_id)
    train_id = all_id[:int(n_nodes*train_rate)]
    val_id = all_id[int(n_nodes*0.5): int(n_nodes*(1+0.5)/2)]
    test_id = all_id[int(n_nodes*(1+0.5)/2): -1]
    train_mask = torch.zeros(n_nodes )
    train_mask [train_id] = 1
    train_mask = train_mask.bool()
    val_mask = torch.zeros(n_nodes )
    val_mask [val_id] = 1
    val_mask = val_mask.bool()
    test_mask = torch.zeros(n_nodes )
    test_mask [test_id] = 1
    test_mask = test_mask.bool()
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask
    # Mask nodes which are labeled in SEMI
    data['train_label'] = data.train_mask&label_mask
    data['val_label'] = data.val_mask&label_mask
    data['test_label'] = data.test_mask&label_mask
    
    return data, 2