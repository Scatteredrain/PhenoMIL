import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss
    
    def confidence_update(self, temp_un_conf, batch_index, batchY, true_labels):
        # print(self.confidence)
        # print(batch_index)
        # print(true_labels)
        with torch.no_grad():
            '''update confidence with CLS-2 (from prototypes) and maybe [true labels]'''
            _, prot_pred = (temp_un_conf * batchY).max(dim=1)
            pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()
            self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :]\
                 + (1 - self.conf_ema_m) * pseudo_label
            
            self.confidence[batch_index, :][true_labels == 0] = torch.tensor([1,0], device=self.confidence.device, dtype=torch.float) \
                if self.num_class == 2 else torch.tensor([1,0,0], device=self.confidence.device, dtype=torch.float)
            self.confidence[batch_index, :][true_labels == 1] = torch.tensor([0,1],  device=self.confidence.device, dtype=torch.float) \
                if self.num_class == 2 else torch.tensor([0,1,0], device=self.confidence.device, dtype=torch.float)
        return None

class partial_loss_DDP(nn.Module):
    def __init__(self, confidence, conf_ema_m=0.99, num_class=2):
        super().__init__()
        self.register_buffer('confidence', confidence)  # 关键：注册为缓冲区
        self.conf_ema_m = conf_ema_m
        self.num_class = num_class
        # self.init_conf = confidence.detach().clone()  # 保存初始值（可选）
        self._device = confidence.device

    def set_conf_ema_m(self, epoch, args):
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1. * epoch / args.epochs * (end - start) + start

    def confidence_update(self, temp_un_conf, batch_index, batchY, true_labels):
        
        with torch.no_grad():
            # --- 步骤1：各GPU独立计算本地更新 ---
            _, prot_pred = (temp_un_conf * batchY).max(dim=1)
            pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().to(self._device)
            
            # 本地更新规则（仅修改当前GPU处理的索引）
            local_conf = self.confidence.data.clone()
            local_conf[batch_index] = self.conf_ema_m * local_conf[batch_index] + (1 - self.conf_ema_m) * pseudo_label
            
            # --- 步骤2：应用真实标签覆盖规则 ---
            override_tensor_0 = torch.tensor([1.0, 0.0] if self.num_class == 2 else [1.0, 0.0, 0.0],
                                            device=self._device)
            override_tensor_1 = torch.tensor([0.0, 1.0] if self.num_class == 2 else [0.0, 1.0, 0.0],
                                            device=self._device)
            local_conf[batch_index[true_labels == 0]] = override_tensor_0
            local_conf[batch_index[true_labels == 1]] = override_tensor_1
            
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
    
    def forward(self, outputs, index):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss


class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning: 
        https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, batch_size=-1):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().to(device)
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

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  #[B]

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