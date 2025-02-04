import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.functional as F

class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, layer_num, device):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        self.layers = layer_num
        
        self.emb1 = nn.Embedding(nnodes, dim) 
        self.emb2 = nn.Embedding(nnodes, dim)

        self.lin1 = nn.ModuleList()
        self.lin2 = nn.ModuleList()
        for i in range(layer_num):
            self.lin1.append(nn.Linear(dim,dim))
            self.lin2.append(nn.Linear(dim,dim))

        self.device = device
        self.k = k 
        self.dim = dim

        

    def forward(self, x, idx):
        
        nodevec1 = self.emb1(idx) 
        nodevec2 = self.emb2(idx)

        adj_set = []

        for i in range(self.layers):
            bs=x.shape[0]

            nodevec1 = torch.tanh(self.lin1[i](nodevec1))
            nodevec2 = torch.tanh(self.lin2[i](nodevec2))
            a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
            adj_cs = F.relu(torch.tanh(a)).repeat(bs,1,1) 

            x_trans=torch.transpose(x,1,2)
            static_adj=torch.bmm(x,x_trans)
            eyes_like=torch.eye(self.nnodes).repeat(bs,1,1).to(self.device)
            eyes_like_inf=eyes_like*1e8
            Adj_static=F.leaky_relu(static_adj-eyes_like_inf)
            Adj_static=F.softmax(Adj_static,dim=-1)
            Adj_static=Adj_static+eyes_like
            Adj_static=torch.tanh(Adj_static) 


            adj0=adj_cs+Adj_static
           
            
        
            mask = torch.zeros(idx.size(0), idx.size(0)).repeat(bs,1,1).to(self.device) 
            mask.fill_(float('0')) 
            s1,t1 = adj0.topk(self.k,2) 
            mask.scatter_(2,t1,s1.fill_(1)) 
            adj = adj0*mask



        return adj

class prop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = F.linear(c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = F.linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device) 
        d = adj.sum(1)
        h = x 
        out = [h]
        a = adj / d.view(-1, 1) 
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)

        ho = torch.cat(out,dim=1)

        ho = self.mlp(ho)

        return ho

class GINConv(nn.Module):
    def __init__(self, in_dim, out_dim, gnn_type, eps=0,train_eps=True):
        super(GINConv, self).__init__()
        
        self.gnn_type = gnn_type
        self.eps = nn.Parameter(torch.zeros(1),requires_grad=True) if train_eps else eps
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )    
    def forward(self, x, A):
        if self.gnn_type =='time':
            agg = torch.einsum('btdc,btw->bwdc',(x,A))
        elif self.gnn_type=='nodes':
            agg = torch.einsum('btdc,bdw->btwc',(x,A))
        elif self.gnn_type=='patch':
            agg = torch.einsum('btnlc,bnw->btwlc',(x,A)) 
        elif self.gnn_type=='pinjie':
            agg=torch.einsum('blnc,bnw->blwc',(x,A))
        elif self.gnn_type=='dots':
            x=x.permute(0,1,3,2,4)
            agg = torch.einsum('bwl,btlnc->btwnc',(A,x)) 
            
        out = (1 + self.eps) * x + agg
        
        return out

class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout,gnn_type,train_eps=True):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        
        self.gin_layers = nn.ModuleList()
        self.gin_layers.append(GINConv(in_dim, hidden_dim,gnn_type,train_eps=train_eps))
        for _ in range(num_layers - 1):
            self.gin_layers.append(GINConv(hidden_dim, hidden_dim,gnn_type,train_eps=train_eps))

        
        self.mlp = nn.Linear(hidden_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x, adj):

        
        for gin_layer in self.gin_layers:
            x = gin_layer(x, adj)
            x = self.act(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)
        return x


class nconv(nn.Module):
    def __init__(self,gnn_type):
        super(nconv,self).__init__()
        self.gnn_type = gnn_type
    def forward(self,x, A):
        if self.gnn_type =='time':
            x = torch.einsum('btdc,tw->bwdc',(x,A))
        elif self.gnn_type=='nodes':
            x = torch.einsum('btdc,dw->btwc',(x,A))
        elif self.gnn_type=='patch':
            x = torch.einsum('btnlc,nw->btwlc',(x,A)) 
        elif self.gnn_type=='dots':
            x=x.permute(0,1,3,2,4)
            x = torch.einsum('wl,btlnc->btwnc',(A,x)) 
            x=x.permute(0,1,3,2,4)
        return x.contiguous()