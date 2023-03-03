from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.typing import Adj, OptTensor, PairTensor, OptPairTensor, Adj, Size
from typing import Callable, Union, Optional
import torch
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Parameter
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from typing import Union, Tuple

from torch import Tensor
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear



class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16, activation = 'softmax'):
        super(Attention, self).__init__()
        self.activation = activation
        if hidden_size != 0:
            self.project = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False)
            )
        else:
            self.project = nn.Sequential(
                nn.Linear(in_size, 1, bias=False)
            )

    def forward(self, z):
        w = self.project(z)
        if self.activation == 'softmax':
            beta = torch.softmax(w, dim=1)
        elif self.activation == 'tanh':
            beta = torch.tanh(w)
        return (beta * z).sum(1), beta
    
class DualLEAConv(MessagePassing):
    r"""The local extremum graph neural network operator from the
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph
    Representations" <https://arxiv.org/abs/1911.07979>`_ paper, which finds
    the importance of nodes with respect to their neighbors using the
    difference operator:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i \cdot \mathbf{\Theta}_1 +
        \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot
        (\mathbf{\Theta}_2 \mathbf{x}_i - \mathbf{\Theta}_3 \mathbf{x}_j)

    where :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to
    target node :obj:`i` (default: :obj:`1`)

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will
            not learn an additive bias. (default: :obj:`True`).
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, bias: bool = True, sub = False, atten_hidden=16 ,activation='softmax', **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sub = sub
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.lin = Linear(in_channels[0], out_channels, bias=bias)
        self.proj = Attention(in_channels[1], atten_hidden, activation)
#         self.lin1 = Linear(in_channels[0], out_channels, bias=bias)
#         self.lin2 = Linear(in_channels[0], out_channels, bias=False)
#         self.lin3 = Linear(in_channels[1], out_channels, bias=bias)
#         self.lin4 = Linear(in_channels[1], out_channels, bias=bias)
#         if sub:
#             self.lin_sub = Linear(out_channels, out_channels, bias=bias)
#         if use_weight:
#             self.weight = torch.nn.Parameter(torch.ones(4), requires_grad = True)
#         else:
#             self.weight = torch.nn.Parameter(torch.ones(4), requires_grad = False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
#         self.lin1.reset_parameters()
#         self.lin2.reset_parameters()
#         self.lin3.reset_parameters()
#         self.lin4.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        src, dst = edge_index
        edge_index_reverse = torch.vstack((dst, src))
        
        if isinstance(x, Tensor):
            x = (x, x, x)

        i = x[0]
        s = x[1]
        t = x[2]
        
        # propagate_type: (a: Tensor, b: Tensor, edge_weight: OptTensor)
        out1 = self.propagate(edge_index, a=i, b=s, edge_weight=edge_weight,
                             size=None)


        out2 = self.propagate(edge_index_reverse, a=i, b=t, edge_weight=edge_weight,
                             size=None)
        out = torch.stack([x[0], out1, out2],  dim=1)
        out, att = self.proj(out)
        out = self.lin(out).squeeze()
        return  out, att

    def message(self, a_i: Tensor, b_j: Tensor,
                edge_weight: OptTensor) -> Tensor:
        out = a_i - b_j
        return out if edge_weight is None else out * edge_weight.view(-1, 1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')



class DualSAGEOConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True, bias: bool = True, use_weight=True, aggr='mean',**kwargs):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.use_weight = use_weight
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin = Linear(in_channels[0], out_channels, bias=bias)
#         self.lin_out = Linear(in_channels[0], out_channels, bias=bias)
#         if self.root_weight:
#             self.lin_r = Linear(in_channels[1], out_channels, bias=False)
#         if use_weight:
#             self.weight = torch.nn.Parameter(torch.ones(3), requires_grad = True)
#         else:
#             self.weight = None
        self.reset_parameters()

    
    def reset_parameters(self):
        self.lin.reset_parameters()
#         self.lin_in.reset_parameters()
#         self.lin_out.reset_parameters()
#         if self.root_weight:
#             self.lin_r.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        src, dst = edge_index
        edge_index_reverse = torch.vstack((dst, src))
        # propagate_type: (x: OptPairTensor)
        out_i = self.propagate(edge_index, x=x, size=size)
        out_o = self.propagate(edge_index_reverse, x=x, size=size)
#         out_i = self.lin_in(out_i)
#         out_o = self.lin_out(out_o)
        x_r = x[1]
#         if self.root_weight and x_r is not None:
#             if self.use_weight:
#                 out = self.weight[0]*self.lin_r(x_r) + self.weight[1]*out_o + self.weight[2]*out_i
#             else:
#                 out = self.lin_r(x_r) + out_o + out_i
        out = self.lin(x_r + out_i + out_o)
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)
class DualSAGEAConv(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, bias: bool = True, sub = False, atten_hidden=16 ,activation='softmax', **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sub = sub
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.lin = Linear(in_channels[0], out_channels, bias=bias)
        self.proj = Attention(in_channels[1], atten_hidden, activation)
#         self.lin1 = Linear(in_channels[0], out_channels, bias=bias)
#         self.lin2 = Linear(in_channels[0], out_channels, bias=False)
#         self.lin3 = Linear(in_channels[1], out_channels, bias=bias)
#         self.lin4 = Linear(in_channels[1], out_channels, bias=bias)
#         if sub:
#             self.lin_sub = Linear(out_channels, out_channels, bias=bias)
#         if use_weight:
#             self.weight = torch.nn.Parameter(torch.ones(4), requires_grad = True)
#         else:
#             self.weight = torch.nn.Parameter(torch.ones(4), requires_grad = False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
#         self.lin1.reset_parameters()
#         self.lin2.reset_parameters()
#         self.lin3.reset_parameters()
#         self.lin4.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        src, dst = edge_index
        edge_index_reverse = torch.vstack((dst, src))
        # propagate_type: (x: OptPairTensor)
        out_i = self.propagate(edge_index, x=x)
        out_o = self.propagate(edge_index_reverse, x=x)
        
        
        out = torch.stack([x[0], out_i, out_o],  dim=1)
        out, att = self.proj(out)
        out = self.lin(out).squeeze()
        return  out, att

        


    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
class DualLEOConv(MessagePassing):
    r"""The local extremum graph neural network operator from the
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph
    Representations" <https://arxiv.org/abs/1911.07979>`_ paper, which finds
    the importance of nodes with respect to their neighbors using the
    difference operator:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i \cdot \mathbf{\Theta}_1 +
        \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot
        (\mathbf{\Theta}_2 \mathbf{x}_i - \mathbf{\Theta}_3 \mathbf{x}_j)

    where :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to
    target node :obj:`i` (default: :obj:`1`)

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will
            not learn an additive bias. (default: :obj:`True`).
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, bias: bool = True, sub = False, use_weight=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sub = sub
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.lin = Linear(in_channels[0], out_channels, bias=bias)
#         self.lin1 = Linear(in_channels[0], out_channels, bias=bias)
#         self.lin2 = Linear(in_channels[0], out_channels, bias=False)
#         self.lin3 = Linear(in_channels[1], out_channels, bias=bias)
#         self.lin4 = Linear(in_channels[1], out_channels, bias=bias)
#         if sub:
#             self.lin_sub = Linear(out_channels, out_channels, bias=bias)
#         if use_weight:
#             self.weight = torch.nn.Parameter(torch.ones(4), requires_grad = True)
#         else:
#             self.weight = torch.nn.Parameter(torch.ones(4), requires_grad = False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
#         self.lin1.reset_parameters()
#         self.lin2.reset_parameters()
#         self.lin3.reset_parameters()
#         self.lin4.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        src, dst = edge_index
        edge_index_reverse = torch.vstack((dst, src))
        
        if isinstance(x, Tensor):
            x = (x, x, x)

        i = x[0]
        s = x[1]
        t = x[2]
        
        # propagate_type: (a: Tensor, b: Tensor, edge_weight: OptTensor)
        out1 = self.propagate(edge_index, a=i, b=s, edge_weight=edge_weight,
                             size=None)


        out2 = self.propagate(edge_index_reverse, a=i, b=t, edge_weight=edge_weight,
                             size=None)

        out = self.lin(x[0] + out1 + out2)
        return  out

        


    def message(self, a_i: Tensor, b_j: Tensor,
                edge_weight: OptTensor) -> Tensor:
        out = a_i - b_j
        return out if edge_weight is None else out * edge_weight.view(-1, 1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


class DualSAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True, bias: bool = True, use_weight=True, aggr='mean',**kwargs):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.use_weight = use_weight
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_in = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_out = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)
        if use_weight:
            self.weight = torch.nn.Parameter(torch.ones(3), requires_grad = True)
        else:
            self.weight = None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        self.lin_out.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        src, dst = edge_index
        edge_index_reverse = torch.vstack((dst, src))
        # propagate_type: (x: OptPairTensor)
        out_i = self.propagate(edge_index, x=x, size=size)
        out_o = self.propagate(edge_index_reverse, x=x, size=size)
        out_i = self.lin_in(out_i)
        out_o = self.lin_out(out_o)
        x_r = x[1]
        if self.root_weight and x_r is not None:
            if self.use_weight:
                out = self.weight[0]*self.lin_r(x_r) + self.weight[1]*out_o + self.weight[2]*out_i
            else:
                out = self.lin_r(x_r) + out_o + out_i
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)
    
class DualCONTRAConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True, bias: bool = True, use_weight=True, aggr='mean',**kwargs):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.use_weight = use_weight
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_c = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)
        if use_weight:
            self.weight = torch.nn.Parameter(torch.ones(2), requires_grad = True)
        else:
            self.weight = None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_c.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        src, dst = edge_index
        edge_index_reverse = torch.vstack((dst, src))
        # propagate_type: (x: OptPairTensor)
        out_i = self.propagate(edge_index, x=x, size=size)
        out_o = self.propagate(edge_index_reverse, x=x, size=size)
        out = self.lin_c(out_i-out_o)
        x_r = x[1]
        if self.root_weight and x_r is not None:
            if self.use_weight:
                out = self.weight[0]*self.lin_r(x_r) + self.weight[1]*(out)
            else:
                out = self.lin_r(x_r) + out
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)    

    


class DualLEConv(MessagePassing):
    r"""The local extremum graph neural network operator from the
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph
    Representations" <https://arxiv.org/abs/1911.07979>`_ paper, which finds
    the importance of nodes with respect to their neighbors using the
    difference operator:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i \cdot \mathbf{\Theta}_1 +
        \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot
        (\mathbf{\Theta}_2 \mathbf{x}_i - \mathbf{\Theta}_3 \mathbf{x}_j)

    where :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to
    target node :obj:`i` (default: :obj:`1`)

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will
            not learn an additive bias. (default: :obj:`True`).
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, bias: bool = True, sub = False, use_weight=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sub = sub
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin1 = Linear(in_channels[0], out_channels, bias=bias)
        self.lin2 = Linear(in_channels[0], out_channels, bias=False)
        self.lin3 = Linear(in_channels[1], out_channels, bias=bias)
        self.lin4 = Linear(in_channels[1], out_channels, bias=bias)
        if sub:
            self.lin_sub = Linear(out_channels, out_channels, bias=bias)
        if use_weight:
            self.weight = torch.nn.Parameter(torch.ones(4), requires_grad = True)
        else:
            self.weight = torch.nn.Parameter(torch.ones(4), requires_grad = False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        src, dst = edge_index
        edge_index_reverse = torch.vstack((dst, src))
        
        if isinstance(x, Tensor):
            x = (x, x, x)

        i = self.lin2(x[0])
        s = self.lin3(x[1])
        t = self.lin4(x[2])
        
        # propagate_type: (a: Tensor, b: Tensor, edge_weight: OptTensor)
        out1 = self.propagate(edge_index, a=i, b=s, edge_weight=edge_weight,
                             size=None)
        out1 = self.weight[1]*out1

        out2 = self.propagate(edge_index_reverse, a=i, b=t, edge_weight=edge_weight,
                             size=None)
        out2 = self.weight[2]*out2
        if self.sub: 
            out3 = out1/edge_index.shape[1] - out2/edge_index_reverse.shape[1]
            
            out3 = self.weight[3]*self.lin_sub(out3)
            return  self.weight[0]*self.lin1(x[0])+out1+out2+out3
        else:
            return  self.weight[0]*self.lin1(x[0])+out1+out2

        


    def message(self, a_i: Tensor, b_j: Tensor,
                edge_weight: OptTensor) -> Tensor:
        out = a_i - b_j
        return out if edge_weight is None else out * edge_weight.view(-1, 1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
    
class DualCATConv(MessagePassing):
    r"""The local extremum graph neural network operator from the
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph
    Representations" <https://arxiv.org/abs/1911.07979>`_ paper, which finds
    the importance of nodes with respect to their neighbors using the
    difference operator:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i \cdot \mathbf{\Theta}_1 +
        \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot
        (\mathbf{\Theta}_2 \mathbf{x}_i - \mathbf{\Theta}_3 \mathbf{x}_j)

    where :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to
    target node :obj:`i` (default: :obj:`1`)

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will
            not learn an additive bias. (default: :obj:`True`).
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_self = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_cat = Linear(in_channels[0]*2, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_self.reset_parameters()
        self.lin_cat.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        src, dst = edge_index
        edge_index_reverse = torch.vstack((dst, src))
        
        if isinstance(x, Tensor):
            x = (x, x, x)

        i = x[0]
        s = x[1]
        t = x[2]
        
        # propagate_type: (a: Tensor, b: Tensor, edge_weight: OptTensor)
        out1 = self.propagate(edge_index, a=i, b=s, edge_weight=edge_weight,
                             size=None)
        out1 = self.lin_cat(out1)
        out2 = self.propagate(edge_index_reverse, a=i, b=t, edge_weight=edge_weight,
                             size=None)
        out2 = self.lin_cat(out2)
        out = self.lin_self(i+ out1 + out2) 
        
        return out
        


    def message(self, a_i: Tensor, b_j: Tensor,
                edge_weight: OptTensor) -> Tensor:
        out = torch.cat((a_i - b_j, b_j), -1)
        return out 

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
    
class DualCATAConv(MessagePassing):
    r"""The local extremum graph neural network operator from the
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph
    Representations" <https://arxiv.org/abs/1911.07979>`_ paper, which finds
    the importance of nodes with respect to their neighbors using the
    difference operator:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i \cdot \mathbf{\Theta}_1 +
        \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot
        (\mathbf{\Theta}_2 \mathbf{x}_i - \mathbf{\Theta}_3 \mathbf{x}_j)

    where :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to
    target node :obj:`i` (default: :obj:`1`)

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will
            not learn an additive bias. (default: :obj:`True`).
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, bias: bool = True, atten_hidden=16, dropout=0.0,aggr='add',**kwargs):
        kwargs.setdefault('aggr',aggr)
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = Attention(in_channels, atten_hidden)
        self.lin = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=True),
                nn.ReLU()
            )
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

#         self.lin_self = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_cat = Linear(in_channels[0]*2, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
#         self.lin_self.reset_parameters()
        self.lin_cat.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        src, dst = edge_index
        edge_index_reverse = torch.vstack((dst, src))
        x = self.lin(x)
        if isinstance(x, Tensor):
            x = (x, x, x)

        i = x[0]
        s = x[1]
        t = x[2]
        
        # propagate_type: (a: Tensor, b: Tensor, edge_weight: OptTensor)
        out1 = self.propagate(edge_index, a=i, b=s, edge_weight=edge_weight,
                             size=None)
        out1 = self.lin_cat(out1)
        out2 = self.propagate(edge_index_reverse, a=i, b=t, edge_weight=edge_weight,
                             size=None)
        out2 = self.lin_cat(out2)
        
        out = torch.stack([i, out1, out2],  dim=1)
        out, att = self.proj(out)    
        out = out.squeeze()
        return out, att
        


    def message(self, a_i: Tensor, b_j: Tensor,
                edge_weight: OptTensor) -> Tensor:
        out = torch.cat((a_i - b_j, b_j), -1)
        return out 

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
