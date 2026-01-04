import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def get_confident_indices(predicted_scores, thre_vec_cls, mask_labeled, pos_conf_per=0.95, neg_conf_per=0.95):
    """
    基于每个类别的二值化阈值，选择正样本中最高的pos_conf_per比例，负样本中最低的neg_conf_per比例
    Args:
        predicted_scores: [B, C] 模型输出的各类别预测概率（0~1之间）
        thre_vec_cls: [C] 每个类别的二值化阈值（如0.5）
        pos_conf_per: 正样本中保留的最高置信度比例（如0.95表示保留前5%）
        neg_conf_per: 负样本中保留的最低置信度比例（如0.95表示保留后5%）
    Returns:
        indices_confident: [B, C] 布尔掩码，True表示该样本-类别对应被选中
    """
    device = predicted_scores.device
    B, C = predicted_scores.shape
    
    # Step 1: 根据阈值划分正负样本掩码
    unlabeled_mask = mask_labeled < 1.0 # [B, C]
    pos_mask = predicted_scores > thre_vec_cls.unsqueeze(0)  # [B, C]
    neg_mask = predicted_scores <= thre_vec_cls.unsqueeze(0) # [B, C]

    # Step 2 替代方案（向量化）
    pos_scores = predicted_scores * pos_mask  # 非正样本位置为0
    neg_scores = predicted_scores * neg_mask + (~neg_mask).float()  # 非负样本位置为1（便于取最小值）

    # 计算每个类别的正样本分位数
    num_pos = (pos_mask * unlabeled_mask).sum(dim=0)  # [C]
    # c = 0
    # print(pos_mask[:, c]*unlabeled_mask[:, c])
    pos_quantiles = torch.stack([
        torch.quantile(pos_scores[pos_mask[:, c]*unlabeled_mask[:, c], c], 1 - pos_conf_per) if num_pos[c] > 0 else torch.tensor(0.0).to(device) 
        for c in range(C)
    ])  # [C]

    # 计算每个类别的负样本分位数
    num_neg = (neg_mask * unlabeled_mask).sum(dim=0)  # [C]
    neg_quantiles = torch.stack([
        torch.quantile(neg_scores[neg_mask[:, c]*unlabeled_mask[:, c], c], neg_conf_per) if num_neg[c] > 0 else torch.tensor(1.0).to(device) 
        for c in range(C)
    ])  # [C]
    
    # 生成最终掩码
    pos_confident = (predicted_scores >= pos_quantiles.unsqueeze(0)) & pos_mask
    neg_confident = (predicted_scores <= neg_quantiles.unsqueeze(0)) & neg_mask
    indices_confident = pos_confident | neg_confident # [B,C]

    return indices_confident

def jaccard_similarity(y1, y2, zero_case=1.0) -> float:
    """
    计算两个二进制标签向量的Jaccard相似度.
    
    Args:
        y1, y2: 一维numpy数组，元素为0或1.
        zero_case: 当两向量全为0时的返回值（默认1.0）.
    
    Returns:
        Jaccard相似度，范围[0,1].
    """
    intersection = torch.sum(torch.logical_and(y1, y2))
    union = torch.sum(torch.logical_or(y1, y2))

    if union == 0:
        return zero_case
    else:
        return (intersection / union).item()

# def compute_jaccard_similarity_matrix(matrix1, matrix2):
#     """
#     计算两个矩阵的Jaccard相似度矩阵.
    
#     Args:
#         matrix1, matrix2: 二维numpy数组.
    
#     Returns:
#         Jaccard相似度矩阵.
#     """
#     similarity_matrix = torch.zeros((matrix1.shape[0], matrix2.shape[0]),dtype=torch.float)

#     for i in range(matrix1.shape[0]):
#         for j in range(matrix2.shape[0]):
#             # print(matrix1[i], matrix2[j])
#             similarity_matrix[i, j] = jaccard_similarity(matrix1[i], matrix2[j])
#             # print(i,j, ':', similarity_matrix[i,j])
#     # print('similarity_matrix:')
#     # print(similarity_matrix[:, -20:])

#     return similarity_matrix


def add_or_classes(matrix1: torch.Tensor, weight = 0.5) -> torch.Tensor:
    assert matrix1.dim() == 2, "Input must be a 2D tensor"

    or_neg = torch.max(matrix1[:, :2], dim=1, keepdim=True).values * weight
    or_pos = torch.max(matrix1[:, 2:8], dim=1, keepdim=True).values * weight

    new_matrix = torch.cat([matrix1, or_neg, or_pos], dim=1)
    return new_matrix


def compute_jaccard_similarity_matrix(matrix1, matrix2, zero_case=1.0, label_enhancing=False, enhancing_weight=0.1):
    """
    计算两个二进制矩阵的 Jaccard 相似度矩阵。
    
    Args:
        matrix1 (torch.Tensor): 形状为 [N1, C] 的二进制矩阵。
        matrix2 (torch.Tensor): 形状为 [N2, C] 的二进制矩阵。
        zero_case (float): 当两行全为 0 时的返回值（默认为 1.0）。
    
    Returns:
        torch.Tensor: 形状为 [N1, N2] 的 Jaccard 相似度矩阵。
    """
    if label_enhancing:
        matrix1 = add_or_classes(matrix1, enhancing_weight)
        matrix2 = add_or_classes(matrix2, enhancing_weight)

        class_weights_inv = torch.tensor([1.0] * (matrix1.size(1) - 2) + [1/enhancing_weight, 1/enhancing_weight], device=matrix1.device).unsqueeze(0)
    else:
        class_weights_inv = 1
        
    matrix1 = matrix1.float()
    matrix2 = matrix2.float()

    intersection = torch.matmul(matrix1*class_weights_inv, matrix2.t())  # [N1, N2]

    sum1 = torch.sum(matrix1, dim=1, keepdim=True)  # [N1, 1]
    sum2 = torch.sum(matrix2, dim=1, keepdim=True)  # [N2, 1]
    union = sum1 + sum2.t() - intersection  # [N1, N2]

    jaccard = torch.where(
        union == 0,
        torch.tensor(zero_case, device=union.device),
        intersection / union
    )
    
    return jaccard

class partial_loss(nn.Module):
    def __init__(self, confidence, conf_ema_m=0.99, num_class=2):
        super().__init__()
        self.confidence = confidence
        self.init_conf = confidence.detach()
        self.conf_ema_m = conf_ema_m
        self.num_class = num_class

    def set_conf_ema_m(self, epoch, args):
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1. * epoch / args.epochs * (end - start) + start

    def forward(self, outputs, index):
        average_loss = F.binary_cross_entropy_with_logits(outputs, self.confidence[index, :])
        return average_loss
    
    def confidence_update(self, temp_un_conf, batch_index, batchY, true_labels):
        # true_labels: [B]; temp_un_conf:[B,C]; batchY: [B,C]
        with torch.no_grad():
            '''update confidence with CLS-2 (from prototypes) and maybe [true labels]'''
            # _, prot_pred = (temp_un_conf * batchY).max(dim=1)
            # pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()
            # self.confidence[batch_index, :][true_labels == 0] = torch.tensor([1,0], device=self.confidence.device, dtype=torch.float) \
            #     if self.num_class == 2 else torch.tensor([1,0,0], device=self.confidence.device, dtype=torch.float)
            # self.confidence[batch_index, :][true_labels == 1] = torch.tensor([0,1],  device=self.confidence.device, dtype=torch.float) \
            #     if self.num_class == 2 else torch.tensor([0,1,0], device=self.confidence.device, dtype=torch.float)
            pseudo_label = (temp_un_conf * batchY).detach()
            self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :]\
                 + (1 - self.conf_ema_m) * pseudo_label
            self.confidence[batch_index, :][true_labels != -1] = batchY[true_labels != -1]
        return None

class partial_loss_DDP(nn.Module):
    def __init__(self, confidence, conf_ema_m=0.99, num_class=2, losstype='v1'):
        super().__init__()
        self.register_buffer('confidence', confidence)  # 关键：注册为缓冲区
        self.conf_ema_m = conf_ema_m
        self.num_class = num_class
        # self.init_conf = confidence.detach().clone()  # 保存初始值（可选）
        if losstype == 'v1':
            self.criterion = nn.BCEWithLogitsLoss()
        elif losstype == 'v2':
            self.criterion = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=1)
        self._device = confidence.device

    def set_conf_ema_m(self, epoch, args):
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1. * epoch / (args.epochs-args.semi_start) * (end - start) + start # smaller

    def confidence_update(self, temp_un_conf, batch_index, batchY, true_labels, type='v1'):
        
        with torch.no_grad():
            # --- 步骤1：各GPU独立计算本地更新 ---
            # pseudo_label = (temp_un_conf * batchY).detach()
            
            # 本地更新规则（仅修改当前GPU处理的索引）
            local_conf = self.confidence.data.clone()
            
            # local_conf[batch_index, :] = self.conf_ema_m * local_conf[batch_index, :] + (1 - self.conf_ema_m) * pseudo_label
            local_conf[batch_index, :][true_labels == -1] = self.conf_ema_m * local_conf[batch_index, :][true_labels == -1] + (1 - self.conf_ema_m) * temp_un_conf[true_labels == -1].detach()
            
            # --- 步骤2：应用真实标签覆盖规则 ---
            local_conf[batch_index, :][true_labels != -1] = batchY[true_labels != -1].float()
            
            # --- 步骤3：跨GPU同步更新（仅覆盖各自负责的索引区域）---
            # 创建更新掩码：标记当前GPU修改的位置
            update_mask = torch.zeros_like(self.confidence, dtype=torch.bool)
            update_mask[batch_index] = True
            
            # 同步掩码和更新值
            world_size = dist.get_world_size()
            all_update_masks = [torch.empty_like(update_mask) for _ in range(world_size)]
            all_update_confs = [torch.empty_like(local_conf) for _ in range(world_size)]
            
            # 收集所有GPU的更新区域和值
            dist.all_gather(all_update_masks, update_mask)
            dist.all_gather(all_update_confs, local_conf)
            
            # 合并更新到全局 confidence
            global_conf = self.confidence.data.clone()
            for mask, conf in zip(all_update_masks, all_update_confs):
                global_conf[mask] = conf[mask]  # 只覆盖被各GPU修改的区域
            
            # --- 步骤4：原子更新全局 confidence ---
            self.confidence.data.copy_(global_conf)

        return None
    
    def forward(self, outputs, index, mask_valid=None):
        # print('pred:')
        # print(outputs.shape)
        # print(outputs)
        # print('label')
        # print(self.confidence[index, :].shape)
        # print(self.confidence[index, :])
        pseudo_labels = self.confidence[index, :] # [B,C]
        pseudo_labels = (pseudo_labels > 0.5).long()

        # if args is not None:
        #     mask_valid = (
        #          (pseudo_labels > torch.clamp(args.pos_conf_thres * args.thre_vec_prot / 0.5, min=0.01, max=0.99).to(device=pseudo_labels.device)) | \
        #         (pseudo_labels < torch.clamp(args.neg_conf_thres * args.thre_vec_prot / 0.5, min=0.01, max=0.99).to(device=pseudo_labels.device))
        #     ).long()

        if mask_valid is not None:
            mask_valid = mask_valid.long()
            # mask_valid = ((pseudo_labels > min(0.99, args.pos_conf_thres * args.thre_vec_prot / 0.5)) | (pseudo_labels < max(0.01, args.neg_conf_thres * (1-args.thre_vec_prot) / 0.5))).long() 
            average_loss = self.criterion(outputs, pseudo_labels, mask_valid)
        else:
            # print(outputs, pseudo_labels)
            average_loss = self.criterion(outputs, pseudo_labels)
        # average_loss = F.binary_cross_entropy_with_logits(outputs, self.confidence[index, :])
        return average_loss

class SupConLoss_multispace(nn.Module):
    """Following Supervised Contrastive Learning: 
        https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features_multispace, masks=None, batch_size=-1):
        # features: NxCxD; mask: CxBxN
        device = (torch.device('cuda')
                  if features_multispace.is_cuda
                  else torch.device('cpu'))
        loss_multi_label = torch.zeros(1).to(device)
        if masks is not None:
            for i in range(features_multispace.shape[1]):
                # SupCon loss (Partial Label Mode)
                mask = masks[i].float().detach().to(device) # BxN
                features = features_multispace[:, i] # NxD
                # compute logits
                anchor_dot_contrast = torch.div(
                    torch.matmul(features[:batch_size], features.T),
                    self.temperature)
                # for numerical stability
                logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                logits = anchor_dot_contrast - logits_max.detach()

                # mask-out self-contrast cases
                logits_mask = torch.scatter(
                    torch.ones_like(mask),
                    1,
                    torch.arange(batch_size).view(-1, 1).to(device),
                    0
                )
                mask = mask * logits_mask
                # # save to csv, if csv exists, append
                # if os.path.exists('/mnt/data/yizhenyu/data/HP识别/workspace/PiCO/snapshots/debug/mask.csv'):
                #     df = pd.read_csv('/mnt/data/yizhenyu/data/HP识别/workspace/PiCO/snapshots/debug/mask.csv')
                #     df = pd.concat([df, pd.DataFrame(mask.cpu().numpy())], axis=1)
                #     df.to_csv('/mnt/data/yizhenyu/data/HP识别/workspace/PiCO/snapshots/debug/mask.csv', index=False)
                # else:
                #     df = pd.DataFrame(mask.cpu().numpy())
                #     df.to_csv('/mnt/data/yizhenyu/data/HP识别/workspace/PiCO/snapshots/debug/mask.csv', index=False)
                # df = pd.DataFrame(mask.cpu().numpy())
                # df.to_csv('/mnt/data/yizhenyu/data/HP识别/workspace/PiCO/snapshots/debug/mask.csv', index=False)
                # np.save(mask.cpu().numpy(), '/mnt/data/yizhenyu/data/HP识别/workspace/PiCO/snapshots/debug/mask.npy')


                # compute log_prob
                exp_logits = torch.exp(logits) * logits_mask  
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12) #[B,N]
            
                # compute mean of log-likelihood over positive
                # print(mask.sum(1))
                mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1e-12)
                # print(mask.sum(1).min(),mask.sum(1).max())

                # loss
                loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
                loss = loss.mean()
                loss_multi_label += loss
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            for i in range(features_multispace.shape[1]):
                # SupCon loss (Partial Label Mode)
                features = features_multispace[:, i]
                q = features[:batch_size]
                k = features[batch_size:batch_size*2]
                queue = features[batch_size*2:]
                l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
                # negative logits: NxK
                l_neg = torch.einsum('nc,kc->nk', [q, queue])
                # logits: Nx(1+K)
                logits = torch.cat([l_pos, l_neg], dim=1)

                # apply temperature
                logits /= self.temperature

                # labels: positive key indicators
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                loss = F.cross_entropy(logits, labels)
                loss_multi_label += loss

        return loss_multi_label / features_multispace.shape[1]

class SupConLoss_One_CL_Space(nn.Module):
    """Following Supervised Contrastive Learning: 
        https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, base_temperature=0.07, losstype='v2', topk=6):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.losstype = losstype
        self.topk = topk

        '''
        v1: multilabel similarities weighting + classical SupCon loss
        v2: multilabel similarities weighting + denominator [all pos logits -> 0]
        v3: multilabel similarities weighting + numerator [Top-k pos logits] 
        v4: multilabel similarities weighting + denominator [other all pos logits -> 0]
        '''

    def forward(self, features, mask=None, batch_size=-1):
        # features: NxD; mask: BxN, range: 0-1, indicates the sims of positive samples
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().to(device) # BxN
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )
            
            if self.losstype == 'v3':

                indices1 = torch.arange(batch_size).unsqueeze(1)             # 前 B 列的对角线索引
                indices2 = indices1 + batch_size                              # 中间 B 列的对角线索引
                indices = torch.cat([indices1, indices2], dim=1)     # 合并索引

                # 填充 0 到两处对角线位置
                diagonal_mask = torch.scatter(
                    torch.ones_like(mask),
                    1, 
                    indices.to(device), 
                    0)              

                retained_mask = mask * (1-diagonal_mask)
                non_diagonal_values = mask * diagonal_mask
                values, indices_topk = torch.topk(non_diagonal_values, self.topk, dim=1)

                # 4. 创建非对角线区域的保留张量
                non_diagonal_result = torch.zeros_like(mask)
                non_diagonal_result.scatter_(1, indices_topk, values)

                # 5. 合并对角线和非对角线的结果
                mask = retained_mask + non_diagonal_result

            mask = mask * logits_mask # anchor and negative are 0s

            # # save to csv, if csv exists, append
            # if os.path.exists('/mnt/data/yizhenyu/data/HP识别/workspace/PiCO/snapshots/debug/mask.csv'):
            #     df = pd.read_csv('/mnt/data/yizhenyu/data/HP识别/workspace/PiCO/snapshots/debug/mask.csv')
            #     df = pd.concat([df, pd.DataFrame(mask.cpu().numpy())], axis=1)
            #     df.to_csv('/mnt/data/yizhenyu/data/HP识别/workspace/PiCO/snapshots/debug/mask.csv', index=False)
            # else:
            #     df = pd.DataFrame(mask.cpu().numpy())
            #     df.to_csv('/mnt/data/yizhenyu/data/HP识别/workspace/PiCO/snapshots/debug/mask.csv', index=False)
            # df = pd.DataFrame(mask.cpu().numpy())
            # df.to_csv('/mnt/data/yizhenyu/data/HP识别/workspace/PiCO/snapshots/debug/mask.csv', index=False)
            # np.save(mask.cpu().numpy(), '/mnt/data/yizhenyu/data/HP识别/workspace/PiCO/snapshots/debug/mask.npy')

            # compute log_prob
            if self.losstype == 'v1' or self.losstype == 'v3':
                exp_logits = torch.exp(logits) * logits_mask # 剔除分母中的anchor与自身的sims
            elif self.losstype == 'v2' or 'v4':
                mask_binary = (mask > 0).float()
                exp_logits = torch.exp(logits) * (1-mask_binary) * logits_mask  #[B,K] only save neg parts ->剔除分母中anchor与自身和其所有正样本的sims
            
            if self.losstype == 'v4':
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + torch.exp(logits) + 1e-12) # may < 0 in v2
                # print('mask', mask[:10, 1])
                # print('anchor pos sims:', torch.exp(logits[:10,1]))
                # print('other pos sims sum:', ((torch.exp(logits) * mask_binary).sum(1, keepdim=True)-torch.exp(logits))[:10,1])
                # print('neg sims sum:', exp_logits.sum(1, keepdim=True)[:10])
            else:
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12) # may < 0 in v2
            


            #log_prob: [B,N]，每一行是一个sample和其anchor的sims比值

            # compute mean of log-likelihood over positive
            # print(mask.sum(1))
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1e-12)
            # print("Mask sum per sample:", mask.sum(1))  # 应确保每个样本至少有1个正样本
            # print("Min/Max mask sum:", mask.sum(1).min(), mask.sum(1).max())

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()

        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size*2]
            queue = features[batch_size*2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss



# refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py
class FocalLoss2d(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)
        
        weight = Variable(self.weight)
           
        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -( (1-pt)**self.gamma ) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip_x_neg=0.05, clip_y_pos=0.05, clip_y_neg=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip_y_pos = clip_y_pos
        self.clip_y_neg = clip_y_neg
        self.clip = clip_x_neg
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y, mask=None):
        """"
        Parameters
        ----------
        x: input logits, [B,C]
        y: targets (multi-label binarized vector), [B,C] 
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1) # [self.clip,1]

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        if mask is not None:
            self.loss *= mask
        #     return -self.loss.sum() / mask.sum()
        # return -self.loss.mean()
        return -self.loss.sum(axis=1).mean()
        





class MultiLabelSoftmax(nn.Module):
    def __init__(self, gamma_pos=1., gamma_neg=1.):
        super(MultiLabelSoftmax, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, outputs, targets):
        '''
        Code referred from "https://github.com/bojone/bert4keras". 
        '''
        targets = targets.float()
        outputs = (1 - 2 * targets) * outputs
        y_pred_neg = outputs - targets * 1e15
        y_pred_pos = outputs - (1 - targets) * 1e15
        zeros = torch.zeros_like(outputs[..., :1]) #[B,1]
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1) #[B,C+1]
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1) #[B,C+1]

        neg_loss = (1 / self.gamma_neg) * torch.log(torch.sum(torch.exp(self.gamma_neg * y_pred_neg), dim=-1))
        pos_loss = (1 / self.gamma_pos) * torch.log(torch.sum(torch.exp(self.gamma_pos * y_pred_pos), dim=-1))

        loss = torch.mean(neg_loss + pos_loss)
        return loss


import numpy as np
from scipy.spatial.distance import pdist, squareform

def jaccard_distance(u, v):
    """Jaccard距离计算"""
    intersection = np.logical_and(u, v).sum()
    union = np.logical_or(u, v).sum()
    return 1 - intersection / union if union != 0 else 1.0

class LabelCluster:
    def __init__(self, id, labels, size):
        self.id = id          # 簇ID（新生成的父节点ID）
        self.labels = labels  # 包含的原始标签索引列表
        self.size = size      # 簇大小（用于平均链接）

def label_enhancement(Y):
    """
    输入:
        Y : numpy数组, 形状为(n_samples, m_labels), 二值标签矩阵
    输出:
        enhanced_Y : 增强后的标签矩阵, 新增m-2个父标签
    """
    n, m = Y.shape
    assert m >= 2, "至少需要2个原始标签"
    
    # --- 阶段1: 层次聚类生成父节点 ---
    # 初始化簇列表，每个原始标签为一个簇
    clusters = [LabelCluster(id=i, labels=[i], size=1) for i in range(m)]
    next_cluster_id = m  # 新生成的父节点ID从m开始
    
    # 存储所有非叶子节点（父节点）及其子标签关系
    parent_children = {}
    
    # 计算初始距离矩阵
    dist_matrix = squareform(pdist(Y.T, metric=jaccard_distance))
    np.fill_diagonal(dist_matrix, np.inf)  # 避免自身比较
    
    # 迭代合并簇，直到只剩一个簇
    while len(clusters) > 1:
        # 找到距离最近的两个簇
        min_dist = np.inf
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = dist_matrix[i][j]
                if dist < min_dist:
                    min_dist = dist
                    a, b = i, j
        
        # 合并簇a和簇b
        cluster_a = clusters[a]
        cluster_b = clusters[b]
        
        # 记录父节点的子标签关系（关键！）
        new_labels = cluster_a.labels + cluster_b.labels
        new_cluster = LabelCluster(
            id=next_cluster_id,
            labels=new_labels,
            size=cluster_a.size + cluster_b.size
        )
        parent_children[next_cluster_id] = {
            'children': [cluster_a.id, cluster_b.id],
            'labels': new_labels
        }
        next_cluster_id += 1
        
        # 更新距离矩阵（平均链接法）
        new_dist_row = []
        for k in range(len(clusters)):
            if k == a or k == b:
                continue
            # 计算新簇与簇k的距离
            d_ak = dist_matrix[a][k]
            d_bk = dist_matrix[b][k]
            new_dist = (cluster_a.size * d_ak + cluster_b.size * d_bk) / (cluster_a.size + cluster_b.size)
            new_dist_row.append(new_dist)
        
        # 移除旧簇a和b，添加新簇
        clusters = [c for idx, c in enumerate(clusters) if idx not in (a, b)]
        clusters.append(new_cluster)
        
        # 更新距离矩阵
        dist_matrix = np.delete(dist_matrix, [a, b], axis=0)
        dist_matrix = np.delete(dist_matrix, [a, b], axis=1)
        new_dist_row = np.array(new_dist_row)
        dist_matrix = np.vstack([dist_matrix, new_dist_row])
        dist_matrix = np.column_stack([dist_matrix, np.append(new_dist_row, np.inf)])
    
    # --- 阶段2: 生成增强标签（m-2个新标签）---
    # 提取所有非叶子节点（排除最后的根节点，共m-2个）
    parent_nodes = list(parent_children.keys())[:m-2]
    
    # 初始化增强标签矩阵
    enhanced_Y = np.zeros((n, m + len(parent_nodes)), dtype=int)
    enhanced_Y[:, :m] = Y
    
    # 为每个父节点生成新标签
    for idx, parent_id in enumerate(parent_nodes):
        children = parent_children[parent_id]['labels']
        # 检查样本是否包含所有子标签
        has_all = np.all(Y[:, children], axis=1)
        enhanced_Y[:, m + idx] = has_all.astype(int)
    
    return enhanced_Y

# # ----------------------
# # 使用示例
# if __name__ == "__main__":
#     # 示例数据: 5个样本，4个原始标签（m=4 → 新增m-2=2个标签）
#     Y = np.array([
#         [1, 0, 1, 0],
#         [0, 1, 1, 0],
#         [1, 1, 0, 1],
#         [0, 1, 0, 1],
#         [1, 0, 0, 0]
#     ], dtype=int)
    
#     enhanced_Y = label_enhancement(Y)
#     print("原始标签矩阵:\n", Y)
#     print("增强后的标签矩阵:\n", enhanced_Y)
#     print("新增标签数量:", enhanced_Y.shape[1] - Y.shape[1])

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # print(pred.max(), pred.min())
    # print(mask.max(), mask.min())

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def attention_constrain_loss(logits_pre, labels, importance_scores_raw):
        
    # 提取分类结果
    logits_pre_neg_b = (torch.max(logits_pre[..., :2], dim=-1)[0] > 0.6).long().detach()  # [B, K]
    logits_pre_pos_b = (torch.max(logits_pre[..., 2:], dim=-1)[0] > 0.6).long().detach() # [B, K]

    # 根据 bag 标签生成条件掩码
    is_negative_bag = labels == 0  # [B]

    # 扩展维度以便广播
    is_negative_bag_expanded = is_negative_bag.unsqueeze(1).to(logits_pre_neg_b.device)  # [B, 1]

    # 定义条件逻辑（通过广播）
    related_mask = (
        (~is_negative_bag_expanded) * logits_pre_pos_b + 
        is_negative_bag_expanded * logits_pre_neg_b
    ).bool()  # [B, K]

    unknown_mask = (
        (~is_negative_bag_expanded) * (logits_pre_neg_b.bool() & ~logits_pre_pos_b.bool()) +
        is_negative_bag_expanded * (logits_pre_pos_b.bool() & ~logits_pre_neg_b.bool())
    ).bool()

    unrelated_mask = (~logits_pre_neg_b.bool() & ~logits_pre_pos_b.bool())  # [B, K]

    if not related_mask.any():
        return torch.tensor(0.0)

    logits = importance_scores_raw[related_mask | unrelated_mask] # [L]
    logits = logits - logits.min()  # 避免负数（可选）
    logits_log_softmax = F.log_softmax(logits, dim=0)

    sum_related = related_mask.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [B, 1]
    # 构造目标分布：related_mask → 1/sum_related，unrelated_mask → 0
    labels = (related_mask.float() / sum_related)[related_mask | unrelated_mask].float() # [L]

    # 计算 KL 散度损失
    loss = F.kl_div(logits_log_softmax, labels, reduction='batchmean')

    return loss

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=2, feat_dim=512, prototypes=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim  = feat_dim
        if prototypes is not None:
            self.centers = nn.Parameter(prototypes)
            # frozen centers
            self.centers.requires_grad = False
        else:
            center_init = torch.zeros(self.num_classes, self.feat_dim).cuda()

            nn.init.xavier_uniform_(center_init)
            self.centers = nn.Parameter(center_init)

        # self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

    def forward(self, x, labels, class_weight=None):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # print('shape', x.shape, labels.shape, self.centers.shape)
        # print('range', self.centers.min(), self.centers.max(), x.min(), x.max())
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long() # should be long()
        classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask   = labels.eq(classes.expand(batch_size, self.num_classes))
        if class_weight is not None:
            class_weight = class_weight.cuda()
            dist = distmat * mask.float() * class_weight[labels.squeeze()]
        else:
            dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        # print('loss',loss)

        return loss

    def get_assignment(self, batch):
        alpha = 1.0
        norm_squared = torch.sum((batch.unsqueeze(1) - self.centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / alpha))
        power = float(alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def target_distribution(self, batch):
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()
    
def One_Bag_Loss(logits_pre, logits_pre_k, labels_instance, labels_bag, criterion1, criterion2, BETA=4):
    """ bag of one instance loss."""
    
    # logits_pre: [B, 2], labels_instance: [B, K], labels_bag: [B]
    # criterion2: torch.nn.KLDivLoss(reduction='none')

    bag_pos_part = (labels_bag == 1)
    bag_neg_part = (labels_bag == 0)

    weight_mask = bag_pos_part * torch.max(labels_instance[..., 2:], dim=-1)[0] + bag_neg_part * torch.max(labels_instance[..., :2], dim=-1)[0]
    weight_mask = weight_mask**BETA

    # loss1 = criterion1(logits_pre, labels_bag)  #[B]
    # loss1 = loss1 * weight_mask 
    loss2 = criterion2(F.log_softmax(logits_pre, dim=-1), F.softmax(logits_pre_k, dim=-1)) #[B, K]
    loss2 = torch.einsum('nc,n->nc', loss2, weight_mask)

    # loss = loss1.mean() + loss2.mean()
    # print(loss1.shape, loss2.shape, weight_mask.shape)

    loss = loss2.mean()
    # loss = loss1.mean()
    # logits_pre_k和labels_bag的差异应该较大，无法共用
    # 但loss1也收敛差，

    # print(logits_pre, logits_pre_k, weight_mask, loss)

    return loss

