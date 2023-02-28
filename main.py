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
import mlflow
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from torch.autograd import Variable


from model import *
from layers import *
from utils import *
from dataloader import *
data_dir='/data/zhihao-sig2-backup/jupyterprojects/Bitcoin'
model_name = 'BTC-2015-Anomaly-rate-new'
mlflow.set_tracking_uri('/data/zhihao-sig2-backup/jupyterprojects/mlruns')
try:
    mlflow.create_experiment(model_name)
except:
    print("Experiment has been created")



def main(args, g, in_sentences, out_sentences, file_name ):
    # full graph training
    profiler = cProfile.Profile()
    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')
    n_classes = 2
    #     g, n_classes = load_pseudo_pg(args.data_name, graph_type=args.graph_type, feature_type=args.feature_type, 
    #                                            use_unlabeled=args.use_unlabeled, scale = args.scale, 
    #                                            train_rate=args.train_rate, random_state=args.random_state)
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
    #            num_workers=0)
    loader_test =  DualNeighborSampler(g.edge_index,
        node_idx = test_nid,
        # Sample 30 neighbors for each node for 2 iterations
        sizes = [25, 10],
        # Use a batch size of 128 for sampling training nodes
        batch_size=int(args.batch_size/2),)
    #        num_workers=0)

    loader_val =  DualNeighborSampler(g.edge_index,
        node_idx = val_nid,
        # Sample 30 neighbors for each node for 2 iterations
        sizes= [25, 10],
        # Use a batch size of 128 for sampling training nodes
        batch_size=int(args.batch_size/2),)





    with mlflow.start_run():
#         profiler.enable()
        mlflow.log_artifact(file_name, artifact_path='code') 
        mlflow.log_params(vars(args))
        best = []
        for repeat in range(1):
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
            model = Binary_Classifier(in_feats, args.num_hidden, args.num_outputs,  args.rnn_in_channels, 
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
                mlflow.log_metrics(params, step=epoch+1)
                mlflow.log_metric('batch-loss', batch_loss, step=epoch+1)
                toc = time.time()
                torch.save(model.cpu(), f'temp/BPT_2015_dual_light_{args.model}_{args.num_layers}.pt')
                mlflow.log_artifact(f'temp/BPT_2015_dual_light_{args.model}_{args.num_layers}.pt', artifact_path='model') 
                if (epoch+1) % log_every_epoch == 0:
                    val_results, test_results = evaluate_light(model, g, loader_val, loader_test, sens_selector,in_sentences, 
                                                               out_sentences, device)
                    mlflow.log_metrics(test_results, step=epoch+1)
                    mlflow.log_metrics(val_results, step=epoch+1)
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
#                         mlflow.log_metrics(best_test)
#                 if (epoch+1) >=  (args.num_epochs-5):
#                     log_every_epoch = 1 
            best.append(list(best_test.values()))
        best = torch.as_tensor(best)
        torch.save(best, f'./results/{args.model}_best_results')
        mlflow.log_artifact(f'./results/{args.model}_best_results', artifact_path='results') 
        best_all = {'Best_Precision_mean':best.mean(0)[0].item(), 
                             'Best_Recall_mean':best.mean(0)[1].item(), 
                             'Best_F1_mean':best.mean(0)[2].item(), 
                             'Best_AUC_mean':best.mean(0)[3].item(),
                   'Best_Precision_std':best.std(0)[0].item(), 
                             'Best_Recall_std':best.std(0)[1].item(), 
                             'Best_F1_std':best.std(0)[2].item(), 
                             'Best_AUC_std':best.std(0)[3].item()}
        mlflow.log_metrics(best_all)
        
        
if __name__ == '__main__':
    needed_dirs = ['./results', './embs', './model', './profilers', './temp']
    for dir_name in needed_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
    file_name = 'BTC_2015_sampling_Multi_light_dual_cat-A-anomaly-rate-light.ipynb'

    sb = 'n'
    length = 32
    in_sentences = np.load(f'/data/zhihao-sig2-backup/jupyterprojects/Bitcoin/data/2015/pre-sampled/in_sentences_{sb}_{length}.npy',allow_pickle=True)
    out_sentences = np.load(f'/data/zhihao-sig2-backup/jupyterprojects/Bitcoin/data/2015/pre-sampled/out_sentences_{sb}_{length}.npy',allow_pickle=True)
    in_sentences_len = torch.load(f'/data/zhihao-sig2-backup/jupyterprojects/Bitcoin/data/2015/pre-sampled/in_sentences_len_{length}.pt')
    out_sentences_len = torch.load(f'/data/zhihao-sig2-backup/jupyterprojects/Bitcoin/data/2015/pre-sampled/out_sentences_len_{length}.pt')
    lens_in = torch.as_tensor(in_sentences_len)
    lens_out = torch.as_tensor(out_sentences_len)
    in_sentences = pad_sequence(in_sentences.tolist(), batch_first=True)
    out_sentences = pad_sequence(out_sentences.tolist(), batch_first=True)
    
    import gc
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default='mlp')
    argparser.add_argument('--data-name', type=str, default='PyG_BTC_2015')
    argparser.add_argument('--use-unlabeled', type=str, default='SEMI', help="Regard unlabeled samples as negative or not.")
    argparser.add_argument('--scale', type=str, default='std')
    argparser.add_argument('--rnn', type=str, default='gru', help="{lstm, gru}")
    argparser.add_argument('--rnn-feat', type=str, default='e', help="{e, nne}")
    argparser.add_argument('--rnn-agg', type=str, default='last', help="{last, max, min, sum,}")
    argparser.add_argument('--lstm-norm', type=str, default='ln', help="{ln, bn, none}")
    argparser.add_argument('--gnn-norm', type=str, default='bn')
    argparser.add_argument('--sort-by', type=str, default='a',
                        help="Sort txes by")
    argparser.add_argument('--length', type=int, default=128,
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
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--num-outputs', type=int, default=2)
    argparser.add_argument('--decoder-layers', type=int, default=1)
    argparser.add_argument('--rnn_in_channels', type=int, default=8)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--random-state', type=int, default=5211)
    argparser.add_argument('--patience', type=int, default=10)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--weight-lr', type=float, default=0.01)
    argparser.add_argument('--dropout', type=float, default=0.2)
    argparser.add_argument('--num-workers', type=int, default=4,
                        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--sample-gpu', action='store_true',
                        help="Perform the sampling process on the GPU. Must have 0 workers.")
    argparser.add_argument('--inductive', action='store_true',
                        help="Inductive learning setting")

    argparser.set_defaults(directed=True)
    mlflow.set_experiment(model_name)
    # model = 'sage'
    model = 'sage'
    ln = 'ln'
    rc = 16
    rf = 'e'
    nl = 2
    dl = 2 
    agg = 'max'
    go = 'no'
    nl = 2
    argparser.set_defaults(aggr='add')
    argparser.set_defaults(reweight=True)
    argparser.set_defaults(undersample=False)
    for ar in [None, None]: 
        argparser.set_defaults(anomaly_rate=ar)
        g, n_classes = load_pseudo_pg(datadir_path='/data/zhihao-sig2-backup/jupyterprojects/Bitcoin/data/2015', use_unlabeled = 'SEMI', scale='minmax', graph_type = 'MultiDi', feature_type ='edge', train_rate=0.5, anomaly_rate= ar,random_state=5211)
        g.lens_in = in_sentences_len.clone()
        g.lens_out = out_sentences_len.clone()
        for wlr in [0.001]:
            for oversample in [0.0]:
                argparser.set_defaults(oversample=oversample)
                for ef in [1]:
                    for do in [0.2]: # , 0.1, 0.2, 0.3, 0.5
                        for rnn in ['gru']: # 'lstm', 
                            for length in [32]: # 256, 512, 1024
                                for ns in [10]:
                                    for rc in [128]:
                                        for scale in ['minmax']: # 'std', 
                                            for model in ['dualcata-tanh-4']: # 'dualcontra-mean', 'dualcontra-weight-mean'
                                                for gn in ['ln']: #, 16]: #, 'bn'
                                                    for cf in [0]:
                                                        args=['--weight-lr', str(wlr),'--sort-by', sb,'--decoder-layers', str(dl), '--graph-op', str(go), '--neighbor-size', str(ns), '--num-layers', str(nl), '--dropout', str(do), '--emb-first', str(ef), '--rnn', rnn, '--length', str(length), '--model', model, '--rnn-agg', agg, '--num-hidden', str(rc), 
                                                                '--concat-feature', str(cf), '--rnn-feat', rf,'--lstm-norm', ln, '--gnn-norm', gn, '--scale', scale]#, '--num-hidden', nh
                                                        arguments = argparser.parse_args(args=args)

                                                        torch.cuda.empty_cache()
                                                        main(arguments, g, in_sentences, out_sentences,file_name )
                                                        del g
