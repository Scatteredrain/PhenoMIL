
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

class HP_neibor_Atten(nn.Module):
    def __init__(self, in_dim=512, out_dim=512):
        super(HP_neibor_Atten, self).__init__()
        self.custom_att = CustomAttention(input_dim=in_dim, weight_params_dim=out_dim)
        self.wv = nn.Linear(in_dim, out_dim)
        # self.neigh = NeighborAggregator(output_dim=1)

    def forward(self, inputs, sparse_adj):
        # inputs: 1xKxIn_dim, sparse_adj: 1xKxK
        # outputs: xl: 1xKxOut_dim
        attention_matrix = self.custom_att(inputs.squeeze(0)).unsqueeze(0)
        # norm_alpha, alpha = self.neigh([attention_matrix, sparse_adj.squeeze(0)]) # [K,K]
        # xl = norm_alpha.unsqueeze(1) * value

        alpha = attention_matrix * sparse_adj # [1,K,K]
        value = self.wv(inputs) #[1,K,E']

        # norm_alpha = F.softmax(alpha, dim=-1)
        # xl = torch.einsum('bkk,bke->bke', norm_alpha, value)

        norm_alpha = alpha.sum(dim=-1) # [1,K]
        norm_alpha = F.softmax(norm_alpha, dim=-1) #[1,K]
        xl = torch.einsum('bk,bke->be', norm_alpha, value) 
        # xl = torch.einsum('bk,bke->bke', norm_alpha, value) 

        return xl, alpha

# class NeighborAggregator(nn.Module):
#     def __init__(self, output_dim):
#         super(NeighborAggregator, self).__init__()
#         self.output_dim = output_dim

#     def forward(self, inputs):
#         # inputs: data_input: KxK, adj_matrix: KxK
#         # outputs: alpha: K
#         data_input, adj_matrix = inputs
#         sparse_data_input = adj_matrix * data_input  # 假设为密集矩阵
#         reduced_sum = sparse_data_input.sum(dim=1)
#         A_raw = reduced_sum.view(-1)
#         alpha = F.softmax(A_raw, dim=0)
#         return alpha, A_raw

class GatedFeatureFusion(nn.Module):
    """
    门控特征融合模块
    输入:
        - xl: 局部增强特征 [N, D]
        - encoder_output: 全局特征 [N, D]
        - alpha: 注意力系数 [N]
    输出:
        - xo: 融合后的特征 [N, D]
    """
    def __init__(self, input_dim):
        super().__init__()
        self.wv = nn.Linear(input_dim, 512)  # 特征变换层

    def forward(self, xl_raw, encoder_output, alpha):
        # 特征变换: xl = WV * X'
        xl = self.wv(xl_raw)  # [N, 512]

        # 门控权重计算
        wei = torch.sigmoid(-xl * alpha.unsqueeze(1))  # [N, 512]
        squared_wei = torch.square(wei)  # [N, 512]

        # 特征融合
        term1 = xl * 2 * squared_wei  # [N, 512]
        term2 = 2 * encoder_output * (1 - squared_wei)  # [N, 512]
        xo = term1 + term2  # [N, 512]
        return xo


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
    


    

class AggregationBlock(nn.Module):
    def __init__(self, in_dim=512, mid_dim=512, final_dim=512, size=[512,512,256], dropout=0.25, feat_dim=128, num_class=8, gate=True, HPmlp=False, HPhead=False):
        super(AggregationBlock, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, mid_dim), nn.LayerNorm(mid_dim), nn.GELU())
        self.attention_net = create_attention_net(mid_dim, size, dropout, gate)
        self.layer = TransLayer(dim=mid_dim)
        self.norm = nn.LayerNorm(mid_dim)
        self.layer_map_to_final = nn.Sequential(nn.Linear(mid_dim, final_dim), nn.LayerNorm(final_dim), nn.GELU())

        self.HP_neibor_Atten = HP_neibor_Atten(in_dim=mid_dim, out_dim=mid_dim)
        if HPmlp:
            self.HP_mlp = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, feat_dim)
            )
        if HPhead:
            self.HP_head = nn.Linear(in_dim, num_class)
        
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
    
    def forward(self, encoder_inputs):
        
        ## 1. Transformer layer
        t = self.fc(encoder_inputs) # 1xKxOut_dim
        t = self.layer(t)

        ## 2. HP space neibor Attention layer
        z_pre = self.HP_mlp(encoder_inputs) # 1xKxProj_dim
        logits_pre = self.HP_head(encoder_inputs) # 1xKxC
        # logits_pre = torch.sigmoid(logits_pre)
        z_norm = F.normalize(z_pre, p=2, dim=-1)  # 归一化
        sim_matrix = torch.matmul(z_norm, z_norm.transpose(1, 2)) # (B=1, K, K)
        label_sim = F.cosine_similarity(logits_pre.unsqueeze(2), logits_pre.unsqueeze(1), dim=-1)  # (B=1, K, K)
        combined_adj_sim = sim_matrix * 0.5 + label_sim * 0.5  
        # print('t', t.shape)  #[B,K,E]
        # print('combined_adj_sim', combined_adj_sim.shape) #[B,K,K]
        h_adj,_ = self.HP_neibor_Atten(t, combined_adj_sim) #[B,K,E]

        ## 3. Bag contrinbution Attention layer
        Atten_contribution, A_raw, output = self.aggregate(t, h_adj, self.norm, self.attention_net)

        output = self.layer_map_to_final(output)
        # print(h.shape, A.shape, A_raw.shape, slide_level.shape, _h.shape)
        # 1xKxin_dim, 1xK, 1xK, 1xfinal_dim

        # return h, A, A_raw, slide_level_logits, _h
        return output, Atten_contribution, logits_pre # 1xfinal_dim, 1xK

class AggregationBlockv2(nn.Module):
    def __init__(self, in_dim=512, mid_dim=512, final_dim=512, size=[512,512,256], dropout=0.25, feat_dim=128, num_class=8, gate=True, HPmlp=False, HPhead=False, MaskDrop_threshold=0.7):
        super(AggregationBlockv2, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, mid_dim), nn.LayerNorm(mid_dim), nn.GELU())
        self.attention_net = create_attention_net(mid_dim, size, dropout, gate)
        self.layer = TransLayer(dim=mid_dim)
        self.norm = nn.LayerNorm(mid_dim)
        self.sigma = 1
        self.layer_map_to_final = nn.Sequential(nn.Linear(mid_dim, final_dim), nn.LayerNorm(final_dim), nn.GELU())
        self.MaskDrop_threshold = MaskDrop_threshold
        self.HP_neibor_Atten = HP_neibor_Atten(in_dim=mid_dim, out_dim=mid_dim)
        if HPmlp:
            self.HP_mlp = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, feat_dim)
            )
        if HPhead:
            self.HP_head = nn.Linear(in_dim, num_class)
        
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
    
    def forward(self, encoder_inputs):
        
        ## 1. Transformer layer + resnet
        t = self.fc(encoder_inputs) # 1xKxOut_dim
        t = self.layer(t) + encoder_inputs

        ## 2. HP space neibor Attention layer
        with torch.no_grad():
            z_pre = self.HP_mlp(encoder_inputs) # 1xKxProj_dim
            logits_pre = self.HP_head(encoder_inputs) # 1xKxC
            logits_pre = torch.sigmoid(logits_pre)
            z_norm = F.normalize(z_pre, p=2, dim=-1)  # [K, E]

        # sim_matrix = torch.matmul(z_norm, z_norm.transpose(1, 2)) # (B=1, K, K)
        # sum_x2 = torch.sum(z_norm ** 2, dim=1)
        # dot_product = torch.mm(z_norm, z_norm.T)
        # squared_distance = sum_x2.unsqueeze(1) + sum_x2.unsqueeze(0) - 2 * dot_product #[K,K]

        # v1
        # cos_sim = torch.mm(z_norm.squeeze(), z_norm.squeeze().T).detach()  # [-1, 1]
        # squared_distance = 2 * (1 - cos_sim)  # 更高效且数值稳定
        # combined_adj_sim = torch.exp(-1.0 * squared_distance / (2 * self.sigma ** 2)).unsqueeze(0)
        # h_adj,_ = self.HP_neibor_Atten(t, combined_adj_sim) #[B,K,E]

        # v2
        z_norm = z_norm.squeeze().detach()
        cos_sim = torch.mm(z_norm, z_norm.t())  # [-1, 1]
        squared_distance = 2 * (1 - cos_sim)  
        the = 1
        combined_adj_sim = torch.exp(-1.0 * squared_distance / (2 * the ** 2)) # [K,K], E{0,1}

        mask = (combined_adj_sim >= self.MaskDrop_threshold).float()  # shape: [n, n]
        # mask = torch.sigmoid(50*(combined_adj_sim-0.7))

        row_sums = torch.sum(combined_adj_sim * mask, dim=1, keepdim=True)  # shape: [n, 1]

        inv_row_sums = 1.0 / (row_sums + 1e-8) # shape: [n, 1]
        mask_drop = mask * inv_row_sums + (1 - mask) * 1.0  # shape: [n, n]
        h_adj, _= self.HP_neibor_Atten(t, mask_drop) #[B,K,E]

        # label_sim = F.cosine_similarity(logits_pre.unsqueeze(2), logits_pre.unsqueeze(1), dim=-1)  # (B=1, K, K)

        # print('t', t.shape)  #[B,K,E]
        # print('combined_adj_sim', combined_adj_sim.shape) #[B,K,K]
        

        ## 3. Bag contribution Attention layer
        Atten_contribution, A_raw, output = self.aggregate(t, h_adj, self.norm, self.attention_net)

        output = self.layer_map_to_final(output)
        # print(h.shape, A.shape, A_raw.shape, slide_level.shape, _h.shape)
        # 1xKxin_dim, 1xK, 1xK, 1xfinal_dim

        # return h, A, A_raw, slide_level_logits, _h
        return output, Atten_contribution, A_raw, logits_pre, z_norm # 1xfinal_dim, 1xK, 1xKxC, KxProj_dim
    

class AggregationBlockv3(nn.Module):
    '''only attention pass_by aggregation'''
    def __init__(self, in_dim=512, mid_dim=512, final_dim=512, size=[512,512,256], dropout=0.25, feat_dim=128, num_class=8, gate=True, HPmlp=False, HPhead=False):
        super(AggregationBlockv3, self).__init__()
        self.attention_net = create_attention_net(mid_dim, size, dropout, gate)
        if HPmlp:
            self.HP_mlp = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, feat_dim)
            )
        if HPhead:
            self.HP_head = nn.Linear(in_dim, num_class)

    def forward(self, encoder_inputs):
        A_raw, output = self.attention_net(encoder_inputs.squeeze(0))

        A_raw = torch.transpose(A_raw, 1, 0) # [1,K]
        A = F.softmax(A_raw, dim=1)

        # print(A.shape, output.shape)
        output = torch.mm(A, output)

        logits_pre = self.HP_head(encoder_inputs) # 1xKxC
        logits_pre = torch.sigmoid(logits_pre)

        z_pre = self.HP_mlp(encoder_inputs) # 1xKxProj_dim
        z_norm = F.normalize(z_pre, p=2, dim=-1)  # [K, E]

        return output, A, A_raw, logits_pre, z_norm

class AggregationBlockv4(nn.Module):
    def __init__(self, in_dim=512, mid_dim=512, final_dim=512, size=[512,512,256], dropout=0.25, feat_dim=128, num_class=8, gate=True, HPmlp=False, HPhead=False, MaskDrop_threshold=0.7):
        super(AggregationBlockv4, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, mid_dim), nn.LayerNorm(mid_dim), nn.GELU())
        self.attention_net = create_attention_net(mid_dim, size, dropout, gate)
        self.sigma = 1
        self.MaskDrop_threshold = MaskDrop_threshold
        if HPmlp:
            self.HP_mlp = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, feat_dim)
            )
        if HPhead:
            self.HP_head = nn.Linear(in_dim, num_class)

    def forward(self, encoder_inputs):

        A_raw, output = self.attention_net(encoder_inputs.squeeze(0))

        A_raw = torch.transpose(A_raw, 1, 0) # [1,K]
        # A = F.softmax(A_raw, dim=1)

        ## 2. HP space neibor Attention layer
        with torch.no_grad():
            z_pre = self.HP_mlp(encoder_inputs) # 1xKxProj_dim
            logits_pre = self.HP_head(encoder_inputs) # 1xKxC
            logits_pre = torch.sigmoid(logits_pre)
            z_norm = F.normalize(z_pre, p=2, dim=-1)  # [K, E]
        z_norm = z_norm.squeeze().detach()
        cos_sim = torch.mm(z_norm, z_norm.t())  # [-1, 1]
        squared_distance = 2 * (1 - cos_sim)  
        the = 1
        combined_adj_sim = torch.exp(-1.0 * squared_distance / (2 * the ** 2)) # [K,K], E{0,1}
        mask = (combined_adj_sim >= self.MaskDrop_threshold).float()  # shape: [n, n]
        # mask = torch.sigmoid(50*(combined_adj_sim-0.7))
        row_sums = torch.sum(combined_adj_sim * mask, dim=1)  # shape: [n]
        A_redun = 1.0 / (row_sums + 1e-8) # shape: [n]

        # print(A.shape, output.shape)
        A = A_raw * A_redun
        A = F.softmax(A, dim=1)
        output = torch.mm(A, output)

        return output, A, A_raw, logits_pre, z_norm


class AggregationBlockv5(nn.Module):
    def __init__(self, in_dim=512, mid_dim=512, final_dim=512, size=[512,512,256], dropout=0.25, feat_dim=128, num_class=8, gate=True, HPmlp=False, HPhead=False, MaskDrop_threshold=0.7):
        super(AggregationBlockv5, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, mid_dim), nn.LayerNorm(mid_dim), nn.GELU())
        self.attention_net = create_attention_net(mid_dim, size, dropout, gate)
        self.sigma = 1
        self.MaskDrop_threshold = MaskDrop_threshold
        if HPmlp:
            self.HP_mlp = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, feat_dim)
            )
        if HPhead:
            self.HP_head = nn.Linear(in_dim, num_class)

    def forward(self, encoder_inputs):

        A_raw, output = self.attention_net(encoder_inputs.squeeze(0))

        A_raw = torch.transpose(A_raw, 1, 0) # [1,K]
        # A = F.softmax(A_raw, dim=1)

        ## 2. HP space neibor Attention layer
        with torch.no_grad():
            z_pre = self.HP_mlp(encoder_inputs) # 1xKxProj_dim
            logits_pre = self.HP_head(encoder_inputs) # 1xKxC
            logits_pre = torch.sigmoid(logits_pre)
            z_norm = F.normalize(z_pre, p=2, dim=-1)  # [K, E]
        z_norm = z_norm.squeeze().detach()
        cos_sim = torch.mm(z_norm, z_norm.t())  # [-1, 1]
        squared_distance = 2 * (1 - cos_sim)  
        the = 1
        combined_adj_sim = torch.exp(-1.0 * squared_distance / (2 * the ** 2)) # [K,K], E{0,1}
        mask = (combined_adj_sim >= self.MaskDrop_threshold).float()  # shape: [n, n]
        # mask = torch.sigmoid(50*(combined_adj_sim-0.7))
        row_sums = torch.sum(combined_adj_sim * mask, dim=1)  # shape: [n]
        A_redun = 1.0 / (row_sums + 1e-8) # shape: [n]; E{0,1}

        # print(A.shape, output.shape)
        A = F.softmax(A_raw, dim=1) #{0,1}
        A = A * A_redun
        output = torch.mm(A, output)

        return output, A, A_raw, logits_pre, z_norm


class AB_MIL(nn.Module):
    def __init__(self, mid_dim=512, size=[512,512,256], dropout=0.25, gate=True):
        super(AB_MIL, self).__init__()
        self.aggregator = create_attention_net(mid_dim, size, dropout, gate)

    def forward(self, x):
        A_raw, output = self.aggregator(x.squeeze(0))

        A_raw = torch.transpose(A_raw, 1, 0) # [1,K]
        A = F.softmax(A_raw, dim=1)

        # print(A.shape, output.shape)
        output = torch.mm(A, output)
        return output, A

class DSMIL(nn.Module):
    def __init__(self, mid_dim=512, cls_num=2):
        super(DSMIL, self).__init__()

        self.fc_dsmil = nn.Sequential(nn.Linear(mid_dim, cls_num))
        self.q_dsmil = nn.Linear(mid_dim, mid_dim)
        self.v_dsmil = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(mid_dim, mid_dim)
        )
        self.fcc_dsmil = nn.Conv1d(2, 2, kernel_size=mid_dim)
    
    def forward(self, x):

        img_feature = x.squeeze(0) #[k,e]
        device = img_feature.device
        instance_pred = self.fc_dsmil(img_feature) #[k,num_cls]
        V = self.v_dsmil(img_feature)
        Q = self.q_dsmil(img_feature).view(img_feature.shape[0], -1)
        _, m_indices = torch.sort(instance_pred, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(img_feature, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K
        q_max = self.q_dsmil(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
        bag_feature = B.view(1, -1) #[1,C*V]
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc_dsmil(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, bag_feature, instance_pred

if __name__ == '__main__':
    model = DSMIL()
    input = torch.randn(10, 512)
    bag_feature, instance_pred = model(input)
    print(bag_feature.shape, instance_pred.shape)