import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Pscan import pscan


class PMambaEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        self.layers = nn.ModuleList([PMambaBlock(configs) for _ in range(configs.e_layers)])
        self.norm_f = RMSNorm(configs.d_model)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        return x

class PMambaBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.mixer = MambaBlock(configs)
        self.norm = RMSNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.configs = configs

    def forward(self, x):
        output = self.mixer(self.norm(x)) 
        output += x
        return output

class MambaBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        self.in_proj = nn.Linear(configs.d_model, 2 * configs.d_ff, bias=configs.bias)
        
        self.x_proj = nn.Linear(configs.d_ff, configs.dt_rank + 2 * configs.d_state + configs.d_ff, bias=False)

        self.dt_proj = nn.Linear(configs.dt_rank, configs.d_ff, bias=True)

        dt_init_std = configs.dt_rank**-0.5 * configs.dt_scale   
        if configs.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif configs.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(configs.d_ff) * (math.log(configs.dt_max) - math.log(configs.dt_min)) + math.log(configs.dt_min)
        ).clamp(min=configs.dt_init_floor) 
        inv_dt = dt + torch.log(-torch.expm1(-dt)) 
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        A = torch.arange(1, configs.d_state + 1, dtype=torch.float32).unsqueeze(0)

        self.A_log = nn.Parameter(torch.log(A))

        self.out_proj = nn.Linear(configs.d_ff, configs.d_model, bias=configs.bias)

    def forward(self, x):
        _, L, _ = x.shape 

        xz = self.in_proj(x) 
        x, z = xz.chunk(2, dim=-1) 

        x = F.silu(x) 
        y = self.ssm(x)

        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) 

        return output
    
    def ssm(self, x):


        A = -torch.exp(self.A_log.float()) 

        deltaBCD = self.x_proj(x) 
        delta, B, C, D = torch.split(deltaBCD, [self.configs.dt_rank, self.configs.d_state, self.configs.d_state, self.configs.d_ff], dim=-1)
        delta = F.softplus(self.dt_proj(delta)) 

        if self.configs.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):

        deltaA = torch.exp(delta.unsqueeze(-1) * A) 
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) 

        BX = deltaB * (x.unsqueeze(-1)) 
        
        hs = pscan(deltaA, BX)
        
        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A) 
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) 
        BX = deltaB * (x.unsqueeze(-1)) 

        h = torch.zeros(x.size(0), self.configs.d_ff, self.configs.d_state, device=deltaA.device) 
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) 
        y = (hs @ C.unsqueeze(-1)).squeeze(3) 

        y = y

        return y
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output