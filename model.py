import dgl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
import torch
from torch_geometric.nn import APPNP, EdgeConv, LEConv,TransformerConv,GCNConv, SGConv, SAGEConv, GATConv, JumpingKnowledge, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from layers import *
class Binary_Classifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, rnn_in_channels, encoder_layer='gcn', decoder='mlp', rnn='gru', rnn_agg = 'last',num_layers=2,
                 decoder_layers = 1, dropout=0.5, bias=True, save_mem=True, use_bn=True, concat_feature=1, emb_first=1, heads=1,lstm_norm='ln', gnn_norm = 'bn', graph_op='',aggr='add'):
        super(Binary_Classifier, self).__init__()
        self.rnn_agg = rnn_agg
        self.rnn = rnn
        self.concat_feature = concat_feature
        self.emb_first = emb_first
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_layers = encoder_layers = num_layers
        self.decoder_layers = decoder_layers
        self.lstm_norm = lstm_norm
        self.gnn_norm = gnn_norm
        self.graph_op = graph_op
        rnn_out_channels = int(hidden_channels/2)
        # Initialize LSTM part
        if emb_first:
            self.lstm_emb_in = nn.Linear(rnn_in_channels, rnn_out_channels)
            self.lstm_emb_out = nn.Linear(rnn_in_channels, rnn_out_channels)
            if self.lstm_norm == 'bn':
                self.lstm_emb_norm_in =  nn.BatchNorm1d(rnn_out_channels)
                self.lstm_emb_norm_out =  nn.BatchNorm1d(rnn_out_channels)
            elif self.lstm_norm == 'ln':
                self.lstm_emb_norm_in =  nn.LayerNorm(rnn_out_channels)
                self.lstm_emb_norm_out =  nn.LayerNorm(rnn_out_channels)
            if rnn == 'lstm':
                self.lstm_in = nn.LSTM(rnn_out_channels, rnn_out_channels)
                self.lstm_out = nn.LSTM(rnn_out_channels, rnn_out_channels)
            elif rnn == 'gru':
                self.lstm_in = nn.GRU(rnn_out_channels, rnn_out_channels)
                self.lstm_out = nn.GRU(rnn_out_channels, rnn_out_channels)
        else:
            if rnn == 'lstm':
                self.lstm_in = nn.LSTM(rnn_in_channels, rnn_out_channels)
                self.lstm_out = nn.LSTM(rnn_in_channels, rnn_out_channels)
            elif rnn == 'gru':
                self.lstm_in = nn.GRU(rnn_in_channels, rnn_out_channels)
                self.lstm_out = nn.GRU(rnn_in_channels, rnn_out_channels)
        # Initialize GNN part
        self.encoder = nn.ModuleList()
        self.encoder_layer = encoder_layer
        use_rnn = 1
        if 'dualcata' in encoder_layer:
            atten_hidden =  encoder_layer.split('-')[-1]
            if atten_hidden.isdigit():
                atten_hidden = int(atten_hidden)
            else:
                atten_hidden = 16
                
            self.encoder.append(
                DualCATAConv(hidden_channels, hidden_channels, bias=bias, atten_hidden=atten_hidden,aggr=aggr))
            for _ in range(encoder_layers-1):
                self.encoder.append(
                    DualCATAConv(hidden_channels, hidden_channels, bias=bias, atten_hidden=atten_hidden,aggr=aggr))
        else:
            raise NameError(f'{encoder_layer} is not implemented!')
        
        # Initialize decoder
        self.decoder = nn.ModuleList()
        for _ in range(decoder_layers-1):
            self.decoder.append(nn.Linear(hidden_channels, hidden_channels))
        self.decoder.append(nn.Linear(hidden_channels, out_channels))
        
        # Initialize other modules
        self.dropout = dropout
        self.activation = F.relu
        # Normalization layer after each encoder layer
        self.bns = nn.ModuleList()
        if self.lstm_norm == 'bn':
            self.lstm_norm_in =  nn.BatchNorm1d(rnn_out_channels)
            self.lstm_norm_out =  nn.BatchNorm1d(rnn_out_channels)
        elif self.lstm_norm == 'ln':
            self.lstm_norm_in =  nn.LayerNorm(rnn_out_channels)
            self.lstm_norm_out =  nn.LayerNorm(rnn_out_channels)
        if self.gnn_norm == 'ln':
            for _ in range(self.encoder_layers):
                self.bns.append(nn.LayerNorm(hidden_channels))
        elif self.gnn_norm == 'bn':
            for _ in range(self.encoder_layers):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
                
    def forward(self, in_pack, out_pack, lens_in, lens_out, edge_index = None, edge_attr = None): 
#         t0 = time.time()
        # generate lstm embeddings
        if self.emb_first:
#             in_pack, lens_in = pad_packed_sequence(in_pack)
#             out_pack, lens_out = pad_packed_sequence(out_pack)
            in_pack = self.lstm_emb_in(in_pack)
            in_pack = self.lstm_emb_norm_in(in_pack)
            out_pack = self.lstm_emb_out(out_pack)
            out_pack = self.lstm_emb_norm_out(out_pack)
#             tpc = time.time()
#             print(in_pack.shape)
            in_pack = pack_padded_sequence(in_pack, lens_in.cpu(), batch_first=True, enforce_sorted=False)
            out_pack = pack_padded_sequence(out_pack, lens_out.cpu(), batch_first=True, enforce_sorted=False)
        if self.rnn_agg == 'last':
            if self.rnn == 'lstm':
                edges_in, (h_in,c_in)  = self.lstm_in(in_pack)
                edges_out, (h_out,c_out)  = self.lstm_out(out_pack)
            elif self.rnn == 'gru':
                edges_in, h_in  = self.lstm_in(in_pack)
                edges_out, h_out  = self.lstm_out(out_pack)
            h_in = h_in.squeeze(0)
            h_out = h_out.squeeze(0)
            if self.lstm_norm != 'none':
                h_in = self.lstm_norm_in(h_in)
                h_out = self.lstm_norm_out(h_out)
            edges_emb = torch.cat([h_in, h_out],1 )
        else:
            edges_in, *_  = self.lstm_in(in_pack)
            edges_out, *_  = self.lstm_out(out_pack)
            edges_in = pad_packed_sequence(edges_in)[0]
            edges_out = pad_packed_sequence(edges_out)[0]
            if self.rnn_agg == 'max':
                edges_in = torch.max(edges_in, dim=0)[0]
                edges_out = torch.max(edges_out, dim=0)[0]
                edges_emb = torch.cat([edges_in, edges_out],1 )
            if self.rnn_agg == 'mean':
                edges_in = torch.mean(edges_in, dim=0)
                edges_out = torch.mean(edges_out, dim=0)
                edges_emb = torch.cat([edges_in, edges_out],1 )
            if self.rnn_agg == 'sum':
                edges_in = torch.sum(edges_in, dim=0)
                edges_out = torch.sum(edges_out, dim=0)
                edges_emb = torch.cat([edges_in, edges_out],1 )
        x = edges_emb 
#         if 'D' in self.graph_op: # MultiDi to Directed
#             edge_index = torch.unique(edge_index.t(), dim=0).t()
#         if 'S' in self.graph_op: # Remove self-loops
#             edge_index, _ = remove_self_loops(edge_index)
#         if 'U' in self.graph_op: # Directed to Undirected
#             edge_index = to_undirected(edge_index)
        # encode
        for i, conv in enumerate(self.encoder):
            if '_e' in self.encoder_layer:
                x, att= conv(x, edge_index, edge_attr)
            elif self.encoder_layer != 'mlp':
                x, att = conv(x, edge_index)
            else:
                x = conv(x)
            if self.gnn_norm  != 'none':
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i == 0:
                firsta = att.clone().detach().cpu()
        gnn_emb = x.clone()
        # decode
        for i, de in enumerate(self.decoder):
            x = de(x)
            if  i != len(self.decoder)-1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.out_channels != 1:
            x = F.log_softmax(x, dim=1)
        return x, firsta
    
    
    
