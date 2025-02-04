import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.functional as F
from models.layer import *
from layers.PMambaEncoder import PMambaEncoder

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        self.max=nn.MaxPool1d(kernel_size=kernel_size,stride=stride,padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size-1 ) // 2, 1) 
        end = x[:, -1:, :].repeat(1, (self.kernel_size-1 ) // 2, 1) 
        x = torch.cat([front, x, end], dim=1) 
        x = self.avg(x.permute(0, 2, 1)) 
        x = x.permute(0, 2, 1) 
        return x
class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean 
        return res, moving_mean

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.seg_len=configs.seg_len
        self.pred_len = configs.pred_len
        self.graph_encs = nn.ModuleList()
        self.enc_layers = configs.e_layers
        self.anti_ood = configs.anti_ood
        for i in range(self.enc_layers):
            self.graph_encs.append(single_scale_gnn(configs=configs))
        self.decompsition=series_decomp(self.seg_len+1)
        self.tred_pred=nn.Linear(self.seq_len,self.pred_len)
        

    def forward(self, x):
        batch_size=x.shape[0]
        if self.anti_ood:
            seq_last = x[:,-1:,:].detach()
            mean_value_per_feature = torch.mean(x, dim=1).unsqueeze(1).detach() 
            x = x - mean_value_per_feature

        seasonal_init, trend_init = self.decompsition(x)
        trend_init_1,trend_init_2=self.decompsition(trend_init)
        seasonal_init=seasonal_init+trend_init_1
        trend_init=trend_init_2
        trend_init=trend_init.permute(0,2,1)
        tred_pred=self.tred_pred(trend_init)
        tred_pred=tred_pred.permute(0,2,1)
        for i in range(1):
            x = self.graph_encs[i](seasonal_init) 

        pred_x=x+tred_pred

        if self.anti_ood:
                pred_x = pred_x  + mean_value_per_feature
        return pred_x 

class single_scale_gnn(nn.Module):
    def __init__(self, configs):
        super(single_scale_gnn, self).__init__()
        self.init_seq_len = configs.seq_len 
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in 
        self.individual = configs.individual 
        self.dropout=configs.dropout
        self.device='cuda:'+str(configs.gpu)
        self.d_model = configs.hidden 
        self.start_linear = nn.Linear(1,self.d_model) 
        self.seg_len=configs.seg_len
        self.seg_num=self.init_seq_len//self.seg_len
        self.num_layers=configs.num_layers 
        self.pj_num=(self.channels*self.init_seq_len)//self.seg_len  
        self.pj_subgraph_size=configs.pj_subgraph_size
        self.scale_subgraph_size=configs.scale_subgraph_size 
        self.gc_layer=configs.gc_layer

        self.idx = torch.arange(self.pj_num).to(self.device)
        self.pj_gc = graph_constructor(self.pj_num, self.pj_subgraph_size, configs.patchvechidden, self.gc_layer,self.device)
        self.pjgin=GIN(self.d_model,self.d_model,self.d_model,self.num_layers,self.dropout,gnn_type='pinjie')
        self.use_pjgcn=1
        self.Linear = nn.Linear(self.init_seq_len, self.pred_len) 
        self.re_linear=nn.Linear(self.d_model,1)

        self.use_multi_scale=configs.use_multi_scale 
        self.scale_seg_len=configs.scale_seg_len 
        self.pj_num_scales=[
            ((self.channels*self.init_seq_len//(2**i))//self.scale_seg_len)
            for i in range(configs.down_sampling_layers+1)
        ]


        self.scale_idxs=[
            torch.arange(self.pj_num_scales[i]).to(self.device)
            for i in range(configs.down_sampling_layers+1)
        ] 
        self.scale_gcs=nn.ModuleList()
        for i in range(configs.down_sampling_layers+1):
            self.scale_gcs.append(graph_constructor(self.pj_num_scales[i],self.scale_subgraph_size,configs.patchvechidden,self.gc_layer,self.device))
            
        
        self.pred_linear=nn.ModuleList()
        for i in range(configs.down_sampling_layers+1):
            self.pred_linear.append(nn.Linear((self.init_seq_len//(2**i)),self.pred_len))
        self.down_sampling_window=configs.down_sampling_window  
        self.down_sampling_layers=configs.down_sampling_layers  

        self.patch_embedding=nn.Linear(self.seg_len,configs.d_model)
        self.re_patch_embedding=nn.Linear(configs.d_model,self.seg_len)
        self.scale_patch_embedding=nn.Linear(self.scale_seg_len,configs.d_model)
        self.scale_patch_embedding_re=nn.Linear(configs.d_model,self.scale_seg_len)
        
        self.encoder = PMambaEncoder(configs)
        self.scale_encoder=PMambaEncoder(configs)

    
    def multi_scale_process_inputs(self, x_enc): 

        down_pool = torch.nn.AvgPool1d(self.down_sampling_window) 
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc

        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))

        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

        x_enc = x_enc_sampling_list

        return x_enc



    def expand_channel(self,x):
        x=x.unsqueeze(-1) 
        x=self.start_linear(x) 
        return x
    def forward(self, x):
        batch_size = x.shape[0]

        if self.use_multi_scale:
            x_list=self.multi_scale_process_inputs(x)
            x_final_learn=[]
            x_final_pred=[]
            for i in range(len(x_list)):
                if i==0:
                    x_in=x_list[i] 
                    x_pinjie_oj=x_in.reshape(batch_size,-1)
                    x_pinjie_oj=x_pinjie_oj.reshape(batch_size,-1,self.pj_num) 
                    x_pinjie_oj_1=x_pinjie_oj.permute(0,2,1) 

                    x_pinjie=self.expand_channel(x_pinjie_oj) 
                    if self.use_pjgcn:
                        pj_adp=self.pj_gc(x_pinjie_oj_1,self.idx) 
                        x_pinjie=self.pjgin(x_pinjie,pj_adp)+x_pinjie
                    x_pinjie=x_pinjie.reshape(batch_size,-1,self.d_model)
                    x_pinjie=x_pinjie.reshape(batch_size,self.init_seq_len,self.channels,self.d_model)
                    x_pinjie=x_pinjie.permute(0,2,1,3)
                    x_patch=x_pinjie.reshape(batch_size,self.channels,self.seg_num,self.seg_len,-1)
                    x_final_learn.append(x_patch)
                else:
                    x_in=x_list[i]
                    x_pinjie_oj=x_in.reshape(batch_size,-1) 
                    x_pinjie_oj=x_pinjie_oj.reshape(batch_size,-1,self.pj_num_scales[i]) 
                    x_pinjie_oj_1=x_pinjie_oj.permute(0,2,1) 

                    x_pinjie=self.expand_channel(x_pinjie_oj)
                    if self.use_pjgcn:
                        pj_adp=self.scale_gcs[i](x_pinjie_oj_1,self.scale_idxs[i]) 
                        x_pinjie=self.pjgin(x_pinjie,pj_adp)+x_pinjie
                    x_pinjie=x_pinjie.reshape(batch_size,-1,self.d_model)
                    x_pinjie=x_pinjie.reshape(batch_size,(self.init_seq_len//(2**i)),self.channels,self.d_model)
                    x_pinjie=x_pinjie.permute(0,2,1,3)
                    x_patch=x_pinjie.reshape(batch_size,self.channels,(self.init_seq_len//(2**i)//self.scale_seg_len),self.scale_seg_len,-1)
                    x_final_learn.append(x_patch)
            for i in range(len(x_list)):
                if i==0:
                    x_patch=x_final_learn[i]
                    x=x_patch
                    x = self.re_linear(x)  

                    x=x.reshape(batch_size,-1,self.init_seq_len)

                    x_slice=x
                    x_slice=x_slice.reshape(x_slice.shape[0]*x_slice.shape[1],self.seg_num,self.seg_len) 
                    x_slice=self.patch_embedding(x_slice) 
                    enc_out = self.encoder(x_slice)
                    enc_out=self.re_patch_embedding(enc_out)
                    enc_out=enc_out.reshape(batch_size,self.channels,self.init_seq_len)

                    x = self.Linear(enc_out) 
                    
                    x=x.permute(0,2,1)
                    x_final_pred.append(x)
                    
                else:
                    x_patch=x_final_learn[i]
                    x=x_patch
                    x = self.re_linear(x)
                    x=x.squeeze(-1)

                    x=x.reshape(batch_size,-1,self.init_seq_len//(2**i))

                    x_slice=x
                    x_slice=x_slice.reshape(x_slice.shape[0]*x_slice.shape[1],(self.init_seq_len//(2**i)//self.scale_seg_len),self.scale_seg_len) #2688,5,144
                    x_slice=self.scale_patch_embedding(x_slice) 
                    enc_out = self.scale_encoder(x_slice)
                    enc_out=self.scale_patch_embedding_re(enc_out)
                    enc_out=enc_out.reshape(batch_size,self.channels,self.init_seq_len//(2**i))



                    x = self.pred_linear[i](enc_out) 
                    
                    x=x.permute(0,2,1)
                    x_final_pred.append(x)

            sum_tensor=sum(x_final_pred)
            mean_tensor=sum_tensor/len(x_final_pred)
            mean_tensor = F.dropout(mean_tensor,p=self.dropout,training=self.training)
            return mean_tensor

        else:
            x_variables = self.expand_channel(x)
            x_pinjie_oj=x.reshape(batch_size,-1) 
            x_pinjie_oj=x_pinjie_oj.reshape(batch_size,-1,self.pj_num) 
            x_pinjie_ori=x_pinjie_ori.reshape(batch_size,self.pj_num,-1) 
            x_pinjie_oj_1=x_pinjie_oj.permute(0,2,1) 

            x_pinjie=self.expand_channel(x_pinjie_oj) 
            if self.use_pjgcn:
                pj_adp=self.pj_gc(x_pinjie_oj_1,self.idx)
                x_pinjie=self.pjgin(x_pinjie,pj_adp)+x_pinjie
            x_pinjie=x_pinjie.reshape(batch_size,-1,self.d_model)
            x_pinjie=x_pinjie.reshape(batch_size,self.init_seq_len,self.channels,self.d_model)
            x_pinjie=x_pinjie.permute(0,2,1,3)
            x_patch=x_pinjie.reshape(batch_size,self.channels,self.seg_num,self.seg_len,-1)

            
            x=x_patch
            x = self.re_linear(x)
            x=x.squeeze(-1)

            x=x.reshape(batch_size,-1,self.init_seq_len)
            
            
            x_slice=x
            x_slice=x_slice.reshape(x_slice.shape[0]*x_slice.shape[1],self.seg_num,self.seg_len) 
            x_slice=self.patch_embedding(x_slice) 
            enc_out = self.encoder(x_slice)
            enc_out=self.re_patch_embedding(enc_out)
            enc_out=enc_out.reshape(batch_size,self.channels,self.init_seq_len)
            

            x = self.Linear(enc_out) 


            x=x.permute(0,2,1)
            x = F.dropout(x,p=self.dropout,training=self.training)
            return x
            
    

