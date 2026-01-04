import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.nn as nn
from nystrom_attention import NystromAttention
import admin_torch
# from .transmil import NystromAttention

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L=1280, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1280, D=256, dropout=0.25, n_classes=2):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))
        
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.residual_attn = admin_torch.as_module(8)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout = 0.3
        )

    def forward(self, x):
        x = self.residual_attn(x,self.attn(self.norm(x)))
        return x

def create_attention_net(feat_in, sz, dropout, gate):
    fc = [nn.Linear(feat_in, sz[1]), nn.GELU()]
    if dropout:
        fc.append(nn.Dropout(dropout))
    if gate:
        attention_net = Attn_Net_Gated(L=sz[1], D=sz[2], dropout=dropout, n_classes=1)
    else:
        attention_net = Attn_Net(L=sz[1], D=sz[2], dropout=dropout, n_classes=1)
    fc.append(attention_net)
    return nn.Sequential(*fc)

class IAMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, final_dim, size, dropout, gate):
        super(IAMBlock, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.attention_net = create_attention_net(out_dim, size, dropout, gate)
        self.layer = TransLayer(dim=out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.layer_map_to_final = nn.Sequential(nn.Linear(out_dim, final_dim), nn.LayerNorm(final_dim), nn.GELU())
        
    def aggregate(self, h, norm_func, attn_net):
        # print(h.shape)
        _h = norm_func(h).squeeze()  # KxOut_dim
        A, _h = attn_net(_h)  # NxK, N=1    
        A = torch.transpose(A, 1, 0)  # KxN
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        result = (torch.mul(h.squeeze().T, A.squeeze())).sum(dim=1).unsqueeze(0) # 1xOut_dim
        slide_level = norm_func(result)
        return A, A_raw, slide_level, _h
    
    def forward(self, h):
        h = self.fc(h) # 1xKxOut_dim
        h = self.layer(h)
        A, A_raw, slide_level, _h = self.aggregate(h, self.norm, self.attention_net)
        slide_level = self.layer_map_to_final(slide_level)
        # print(h.shape, A.shape, A_raw.shape, slide_level.shape, _h.shape)
        # 1xKxin_dim, 1xK, 1xK, 1xfinal_dim

        # return h, A, A_raw, slide_level_logits, _h
        return slide_level, A, A_raw # 1xfinal_dim, 1xK

class AggregationBlockv2(nn.Module):
    def __init__(self, in_dim=512, mid_dim=512, final_dim=512, size=[512,512,256], dropout=0.25, feat_dim=128, num_class=8, gate=True):
        super(AggregationBlockv2, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, mid_dim), nn.LayerNorm(mid_dim), nn.GELU())
        self.attention_net = create_attention_net(mid_dim, size, dropout, gate)
        self.layer = TransLayer(dim=mid_dim)
        self.norm = nn.LayerNorm(mid_dim)
        self.sigma = 1
        self.layer_map_to_final = nn.Sequential(nn.Linear(mid_dim, final_dim), nn.LayerNorm(final_dim), nn.GELU())

        self.HP_neibor_Atten = HP_neibor_Atten(in_dim=mid_dim, out_dim=mid_dim)
        
    def aggregate(self, t, h_adj, norm_func, attn_net):
        _h = norm_func(t).squeeze()  # KxOut_dim
        A, _h = attn_net(_h)  # Kxn_classes, n_classes=1
        # print('2',A.shape)
        A = torch.transpose(A, 1, 0)  # 1xK
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over K

        result = (torch.mul((t+h_adj).squeeze().T, A.squeeze())).sum(dim=1).unsqueeze(0) # 1xOut_dim
        output = norm_func(result)
        return A, A_raw, output,
    
    def forward(self, encoder_inputs, z_norm):
        
        ## 1. Transformer layer + resnet
        t = self.fc(encoder_inputs) # 1xKxOut_dim
        t = self.layer(t) + encoder_inputs

        ## 2. HP space neibor Attention layer

        # v2
        z_norm = z_norm.squeeze().detach()
        cos_sim = torch.mm(z_norm, z_norm.t())  # [-1, 1]
        squared_distance = 2 * (1 - cos_sim)  
        combined_adj_sim = torch.exp(-1.0 * squared_distance / (2 * 1 ** 2)) # [K,K]

        mask = (combined_adj_sim >= 0.7).float()  # shape: [n, n]
        # mask = torch.sigmoid(50*(combined_adj_sim-0.7))

        row_sums = torch.sum(combined_adj_sim * mask, dim=1, keepdim=True)  # shape: [n, 1]

        inv_row_sums = 1.0 / (row_sums + 1e-8) # shape: [n, 1]
        mask_drop = mask * inv_row_sums + (1 - mask) * 1.0  # shape: [n, n]
        h_adj,_ = self.HP_neibor_Atten(t, mask_drop) #[B,K,E]

        # label_sim = F.cosine_similarity(logits_pre.unsqueeze(2), logits_pre.unsqueeze(1), dim=-1)  # (B=1, K, K)

        # print('t', t.shape)  #[B,K,E]
        # print('combined_adj_sim', combined_adj_sim.shape) #[B,K,K]
        

        ## 3. Bag contribution Attention layer
        Atten_contribution, A_raw, output = self.aggregate(t, h_adj, self.norm, self.attention_net)

        output = self.layer_map_to_final(output)
        # print(h.shape, A.shape, A_raw.shape, slide_level.shape, _h.shape)
        # 1xKxin_dim, 1xK, 1xK, 1xfinal_dim

        # return h, A, A_raw, slide_level_logits, _h
        return output, Atten_contribution, A_raw # 1xfinal_dim, 1xK, 1xKxC, KxProj_dim

class HP_neibor_Atten(nn.Module):
    def __init__(self, in_dim=512, out_dim=512):
        super(HP_neibor_Atten, self).__init__()
        self.custom_att = CustomAttention(input_dim=in_dim, weight_params_dim=out_dim)
        self.wv = nn.Linear(in_dim, out_dim)

    def forward(self, inputs, sparse_adj):
        # inputs: 1xKxIn_dim, sparse_adj: 1xKxK
        # outputs: xl: 1xKxOut_dim
        attention_matrix = self.custom_att(inputs.squeeze(0)).unsqueeze(0)
        # norm_alpha, alpha = self.neigh([attention_matrix, sparse_adj.squeeze(0)]) # [K,K]
        # xl = norm_alpha.unsqueeze(1) * value

        alpha = attention_matrix * sparse_adj # [1,K,K]
        norm_alpha = F.softmax(alpha, dim=-1)
        value = self.wv(inputs) #[1,K,E']
        xl = torch.einsum('bkk,bke->bke', norm_alpha, value)

        return xl, alpha


class CustomAttention(nn.Module):
    def __init__(self, input_dim, weight_params_dim):
        super(CustomAttention, self).__init__()
        self.weight_params_dim = weight_params_dim

        self.wq = nn.Parameter(torch.Tensor(input_dim, weight_params_dim))
        self.wk = nn.Parameter(torch.Tensor(input_dim, weight_params_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.wq)
        nn.init.xavier_uniform_(self.wk)

    def forward(self, inputs):
        q = torch.matmul(inputs, self.wq)
        k = torch.matmul(inputs, self.wk)
        dk = self.weight_params_dim
        matmul_qk = torch.matmul(q, k.t())
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))
        return scaled_attention_logits