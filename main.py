import pickle
import os
from torch_geometric.nn import LabelPropagation
import cProfile, pstats
import os
import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
import torch
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader
import random
from torch.autograd import Variable


from model import *
from layers import *
from utils import *
from dataloader import *




def main(args):
    # Load data
    data_path=f'/data/{args.data}'
    data_path = os.path.join(data_dir, 'data.pt')
    g, n_classes = load_data(data_path,  train_rate=args.train_rate, anomaly_rate= args.anomaly_rate,random_state=args.random_state)
    length = args.length
    sb = 'a'
    in_sentences = np.load(os.paht.join(data_dir, f'/in_sentences_{length}.npy'),allow_pickle=True)
    out_sentences = np.load(os.paht.join(data_dir, f'/out_sentences_{length}.npy'),allow_pickle=True)
    in_sentences_len = torch.load(os.paht.join(data_dir, f'/in_sentences_len_{length}.npy'))
    out_sentences_len = torch.load(os.paht.join(data_dir, f'/out_sentences_len_{length}.npy'))
    lens_in = torch.as_tensor(in_sentences_len)
    lens_out = torch.as_tensor(out_sentences_len)
    in_sentences = pad_sequence(in_sentences.tolist(), batch_first=True)
    out_sentences = pad_sequence(out_sentences.tolist(), batch_first=True)    
    g.lens_in = in_sentences_len.clone()
    g.lens_out = out_sentences_len.clone()
    
    rnn_in_channels = out_sentences.size(-1)
    
    # full graph training
    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')
    n_classes = 2
    
    g.num_nodes = len(g.labels)
    g_train = subdata(g, g.train_mask, relabel_nodes = True)
    g_train.num_nodes = len(g_train.labels)
    in_sentences_train = in_sentences[g.train_mask]
    out_sentences_train = out_sentences[g.train_mask]
    sens_selector_train = PreSentences_light(train=True, train_mask = g.train_mask)
    sens_selector = PreSentences_light()
    in_feats = 0
    train_nid = torch.nonzero(g_train.train_label, as_tuple=True)[0]
    if args.oversample and args.oversample > args.anomaly_rate:
        from imblearn.over_sampling import RandomOverSampler
        oversample = RandomOverSampler(sampling_strategy=args.oversample, random_state=args.random_state)
        nid_resampled, _ = oversample.fit_resample(train_nid.reshape(-1,1),g_train.labels[g_train.train_label])
        train_nid = torch.as_tensor(nid_resampled.reshape(-1))
    val_nid = torch.nonzero(g.val_label, as_tuple=True)[0]
    test_nid = torch.nonzero(g.test_label, as_tuple=True)[0]
    if args.undersample:
        sampler = BalancedSampler(g_train.labels[train_nid])
    else:
        sampler = None
    val_nid = torch.nonzero(g.val_label, as_tuple=True)[0]
    test_nid = torch.nonzero(g.test_label, as_tuple=True)[0]
    loader_train =  DualNeighborSampler(g_train.edge_index,
            sizes = [25,10],
            node_idx = train_nid,
            batch_size=args.batch_size,
            sampler = sampler,
            shuffle = None if sampler else True,)
    loader_test =  DualNeighborSampler(g.edge_index,
        node_idx = test_nid,
        # Sample 30 neighbors for each node for 2 iterations
        sizes = [25, 10],
        # Use a batch size of 128 for sampling training nodes
        batch_size=int(args.batch_size/2),)

    loader_val =  DualNeighborSampler(g.edge_index,
        node_idx = val_nid,
        # Sample 30 neighbors for each node for 2 iterations
        sizes= [25, 10],
        # Use a batch size of 128 for sampling training nodes
        batch_size=int(args.batch_size/2),)

    best = []
    for repeat in range(5):
            # Training loop
        avg = 0
        iter_tput = []
        pred_pos = []
        batch_rate = []
        log_every_epoch = 5
        best_val = 0.0
        best_test = {f'Best_Precision':0, 
                                f'Best_Recall':0, 
                                f'Best_F1':0, 
                                f'Best_AUC':0.5}
        model = Binary_Classifier(in_feats, args.num_hidden, args.num_outputs,  rnn_in_channels, 
                                    num_layers=args.num_layers, rnn_agg = args.rnn_agg, encoder_layer=args.model, concat_feature = args.concat_feature, 
                                    dropout=args.dropout, emb_first = args.emb_first, gnn_norm=args.gnn_norm, 
                                    lstm_norm=args.lstm_norm, graph_op = args.graph_op, decoder_layers=args.decoder_layers,aggr=args.aggr)


        if args.reweight:
            weights = torch.FloatTensor([1/(g_train.labels==0).sum().item(),1/(g_train.labels==1).sum().item()]).to(device)
            print('reweighted')
        else:
            weights = None
        loss_fcn = nn.CrossEntropyLoss(weight =weights)
        weight_params = []
        for pname, p in model.encoder.named_parameters():
            if 'proj' in pname or 'adp' in pname:
                weight_params += [p]
        all_params = model.parameters()
        params_id = list(map(id, weight_params))
        other_params = list(filter(lambda p: id(p) not in params_id, all_params))
        optimizer = optim.Adam([
                {'params': other_params, 'lr': args.lr}, 
                {'params': weight_params, 'lr': args.weight_lr}])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                        milestones=[10, 15], gamma=0.5)
        for epoch in range(args.num_epochs):   
            model = model.to(device)
            model.train()
            batch_loss = 0
            tic = time.time()
            nodes_num = 0
            batch_len = len(loader_train)
#                 att_in = []
#                 att_out = []
            fa = []
            for loader_id, (sub_graph, subset,  batch_size) in enumerate(loader_train):
                sub_graph = sub_graph.to(device)
                in_pack, lens_in = sens_selector_train.select(subset, in_sentences_train, g_train.lens_in)
                out_pack, lens_out = sens_selector_train.select(subset,  out_sentences_train, g_train.lens_out)
#                     edge_attr = g_train.edge_attr[eids].to(device)
                in_pack = in_pack.to(device)
                out_pack = out_pack.to(device)
                f_t0 = time.time()
                batch_pred, a = model( in_pack, out_pack, lens_in, lens_out, sub_graph)
                loss = loss_fcn(batch_pred[:batch_size], g_train.labels[subset][:batch_size].to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                tic_step = time.time()
                batch_loss += loss.item()
                nodes_num += batch_size
                print(f'Epoch {epoch} | Batch {loader_id+1}/{batch_len} | Loss {loss.item()} | Nodes {batch_size}:{len(subset)}')
                fa.append(a)
            # Log parameter weight
            fa = torch.cat(fa, 0)
            fa_mean = fa.mean(0).reshape(-1)
            fa_std = fa.std(0).reshape(-1)
            assert len(fa_mean) == 3 and len(fa_std) == 3
            params = {}
            for i in range(3):
                params[f'{repeat}mean_{i}'] = fa_mean[i].item()
                params[f'{repeat}std_{i}'] = fa_std[i].item()
            toc = time.time()
            if (epoch+1) % log_every_epoch == 0:
                val_results, test_results = evaluate_light(model, g, loader_val, loader_test, sens_selector,in_sentences, 
                                                            out_sentences, device)
                results_str = ""
                for item, value in test_results.items():
                    results_str += item
                    results_str += ':'
                    results_str += str(value)
                    results_str += ''
                print(results_str)
                results_str = ""
                for item, value in val_results.items():
                    results_str += item
                    results_str += ':'
                    results_str += str(value)
                    results_str += ''
                print(results_str)
                if val_results[f'F1_val'] > best_val:
                    best_val = val_results[f'F1_val']
                    test_values = list(test_results.values())
                    for i, (item, value) in enumerate(best_test.items()):
                        best_test[item] = test_values[i]

        best.append(list(best_test.values()))
    best = torch.as_tensor(best)
    best_all = {'Best_Precision_mean':best.mean(0)[0].item(), 
                            'Best_Recall_mean':best.mean(0)[1].item(), 
                            'Best_F1_mean':best.mean(0)[2].item(), 
                            'Best_AUC_mean':best.mean(0)[3].item(),
                'Best_Precision_std':best.std(0)[0].item(), 
                            'Best_Recall_std':best.std(0)[1].item(), 
                            'Best_F1_std':best.std(0)[2].item(), 
                            'Best_AUC_std':best.std(0)[3].item()}
    print(best_all)
        
        
if __name__ == '__main__':
    needed_dirs = ['./results', './embs', './model', './profilers', './temp']
    for dir_name in needed_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            

    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data', type=str, default='EthereumS')
    argparser.add_argument('--model', type=str, default='dualcata-tanh-4')
    argparser.add_argument('--data-name', type=str, default='PyG_BTC_2015')
    argparser.add_argument('--use-unlabeled', type=str, default='SEMI', help="Regard unlabeled samples as negative or not.")
    argparser.add_argument('--scale', type=str, default='minmax')
    argparser.add_argument('--rnn', type=str, default='gru', help="{lstm, gru}")
    argparser.add_argument('--rnn-feat', type=str, default='e', help="{e, nne}")
    argparser.add_argument('--rnn-agg', type=str, default='max', help="{last, max, min, sum,}")
    argparser.add_argument('--lstm-norm', type=str, default='ln', help="{ln, bn, none}")
    argparser.add_argument('--gnn-norm', type=str, default='bn')
    argparser.add_argument('--sort-by', type=str, default='a',
                        help="Sort txes by")
    argparser.add_argument('--length', type=int, default=32,
                        help="the maxiumu length of sentences")
    argparser.add_argument('--gpu', type=int, default=0,
                        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--emb-first', type=int, default=1,
                        help="Whether to embeds the input of RNN first")
    argparser.add_argument('--concat-feature', type=int, default=0)
    argparser.add_argument('--train-rate', type=float, default=0.5)
    argparser.add_argument('--num-epochs', type=int, default=10)
    argparser.add_argument('--batch-size', type=int, default=128)
    argparser.add_argument('--graph-op', type=str, default=None)
    argparser.add_argument('--graph-type', type=str, default='MultiDi')
    argparser.add_argument('--feature-type', type=str, default='node')
    argparser.add_argument('--neighbor-size', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=128)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--num-outputs', type=int, default=2)
    argparser.add_argument('--decoder-layers', type=int, default=2)
    argparser.add_argument('--rnn_in_channels', type=int, default=8)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--random-state', type=int, default=5211)
    argparser.add_argument('--patience', type=int, default=10)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--weight-lr', type=float, default=0.001)
    argparser.add_argument('--dropout', type=float, default=0.2)
    argparser.add_argument('--anomaly_rate', type=float, default=None)
    argparser.add_argument('--reweight', type=bool, default=False)
    argparser.add_argument('--undersample', type=bool, default=False)
    argparser.add_argument('--oversample', type=float, default=None)
    argparser.add_argument('--num-workers', type=int, default=16,
                        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--sample-gpu', action='store_true',
                        help="Perform the sampling process on the GPU. Must have 0 workers.")
    argparser.add_argument('--inductive', action='store_true',
                        help="Inductive learning setting")

    argparser.set_defaults(directed=True)
    argparser.set_defaults(aggr='add')
    

    arguments = argparser.parse_args()
    main(arguments)
