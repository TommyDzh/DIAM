import numpy as np
import torch
from torch_geometric.utils import add_self_loops, remove_self_loops
import pickle 
import networkx as nx
import json
import os
import argparse
from sklearn import preprocessing
dir='./data'
data_dir='./data'
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

def generate_sequences(args):
    needed_dirs = ['./data', f'./data/{args.data}']
    for dir_name in needed_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)    
    # Load data
    datadir_path=f'./data/{args.data}'
    data = torch.load(os.path.join(datadir_path, 'data.pt'))
    # 
    edges_index = data.edge_index
    edge_attr = data.edge_attr.clone() # The edge attr is minmax scaled, the last dim of edge_attr is timestamp
    n_nodes = len(data.y)
    # Add self-loops
    edges_index, edge_attr = remove_self_loops(edges_index, edge_attr)
    start_time = edge_attr[:, -1].min().numpy()
    edge_attr[:, -1] = edge_attr[:, -1] - start_time # Set timestamp start from 0
    edges_index, edge_attr = add_self_loops(edges_index, edge_attr, 0.0) # Add self edge, and set edge values as 0.0
    edges = edges_index.clone().t().numpy()
    # Minmax scaler edge attribute
    scaler_e = preprocessing.MinMaxScaler()
    edge_attr = scaler_e.fit_transform(edge_attr)

    # Construct edge list for each node:  sentence node_id->[eids]
    print("Constructing edge list......")
    out_sentences = {node_id:[] for node_id in range(n_nodes)}
    out_sentences_eid = {node_id:[] for node_id in range(n_nodes)}
    in_sentences = {node_id:[] for node_id in range(n_nodes)}
    in_sentences_eid = {node_id:[] for node_id in range(n_nodes)}
    for eid in range(len(edges)):
        out_sentences[edges[eid][0]].append(edge_attr[eid])
        out_sentences_eid[edges[eid][0]].append(eid) 
        in_sentences[edges[eid][1]].append(edge_attr[eid])
        in_sentences_eid[edges[eid][1]].append(eid) 
    print("Finished constructing edge list！")

    # Construct edge attribute sequences, and sentence length
    print("Constructing edge attribute sequences......")
    sort_dim = args.sort_by 
    length = args.length
    res_in = []
    res_out = []
    for i in range(len(in_sentences)):
        idx_value = torch.FloatTensor(in_sentences[i])
        idx_index = torch.sort(torch.sort(idx_value[:,sort_dim], dim=0, descending=True).indices[:length]).values
        idx_value = idx_value[idx_index]  
        res_in.append(idx_value)
        idx_value = torch.FloatTensor(out_sentences[i])
        idx_index = torch.sort(torch.sort(idx_value[:,sort_dim], dim=0, descending=True).indices[:length]).values
        idx_value = idx_value[idx_index]  
        res_out.append(idx_value)
    res_in = np.array(res_in, dtype=object)
    res_out = np.array(res_out, dtype=object)
    lens_in = []
    lens_out = []
    for s_in, s_out in zip(res_in, res_out):
        lens_in.append(len(s_in))
        lens_out.append(len(s_out))
    lens_in = torch.as_tensor(lens_in)
    lens_out = torch.as_tensor(lens_out)
    print("Finished edge attribute sequences!")
    # Save
    print("Saving data...")
    np.save(f'./data/{args.data}/out_sentences_{length}.npy', res_out)
    np.save(f'./data/{args.data}/in_sentences_{length}.npy', res_in)
    torch.save(lens_out, f'./data/{args.data}/out_sentences_len_{length}.pt')
    torch.save(lens_in, f'./data/{args.data}/in_sentences_len_{length}.pt')            
    print("Done!")
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data', type=str, default='EthereumS')
    argparser.add_argument('--sort-by', type=int, default=-1,
                        help="Sort txes by which attribute dim")
    argparser.add_argument('--length', type=int, default=32,
                        help="the maxiumu length of sentences")

    arguments = argparser.parse_args()
    generate_sequences(arguments)