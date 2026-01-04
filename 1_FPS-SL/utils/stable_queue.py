import torch
import numpy as np
from collections import namedtuple
from utils.endoscopy_HP import PolypDataset_instances
import yaml
import matplotlib.pyplot as plt
import numpy as np
import argparse

class StableDynamicQueue(torch.nn.Module):
    def __init__(self, queue_size=256, buffer_size=32, num_classes=8, 
                 feat_dim=128, alpha=0.3, smooth_factor=0.9, min_ratio=0.03,
                 balance_thresh=0.1, stable_window=5):
        super().__init__()
        # 新增平衡态控制参数
        self.balance_thresh = balance_thresh  # 平衡判断阈值(5%差异)
        self.stable_window = stable_window    # 连续稳定周期数
        self.stable_counter = 0               # 当前连续稳定计数
        self.in_stable_mode = False           # 是否进入稳定模式
        
        # 时间衰减基数调整
        self.decay_base = 0.1  # 从原0.03调整为0.1以加速旧样本淘汰

        # 初始化队列参数
        self.queue_size = queue_size
        self.num_classes = num_classes
        
        # 统计参数
        self.label_counts = torch.zeros(num_classes)
        self.target_ratio = torch.ones(num_classes) / num_classes  # 均匀目标分布
        self.queue_entry_batch = -torch.ones(queue_size,)
        
        # 稳定性增强参数
        self.smooth_factor = smooth_factor   # 滑动平均因子
        self.min_ratio = min_ratio           # 最小类别保留比例
        self.smoothed_counts = None          # 平滑后的类别计数
        
        # 缓冲区设置
        self.buffer_max = buffer_size     # 缓冲区容量
        # 使用register_buffer注册持久缓冲区
        self.register_buffer('buffer_keys', torch.empty(0, feat_dim))
        self.register_buffer('buffer_labels', torch.empty(0, num_classes))
        self.buffer_entry_batch = torch.empty(0, dtype=torch.long)

        self.global_batch_counter = 0
        self.alpha = alpha

    @property
    def buffer(self):
        # 提供字典视图的统一访问接口
        return {
            'keys': self.buffer_keys,
            'labels': self.buffer_labels,
            'entry_batch': self.buffer_entry_batch
        }

    def enqueue_batch(self, queue_ptr, queue, queue_pseudo, features, labels, start_direct_enqueue=False):
        """改进的入队逻辑"""
        # print('3, enqueue')
        current_batch = torch.full((features.size(0),), self.global_batch_counter)
        self.global_batch_counter += 1

        # 首次填充队列时不进行淘汰
        if int(queue_ptr) < self.queue_size:
            ptr = int(queue_ptr)
            valid_size = min(features.size(0), self.queue_size - ptr)
            
            queue[ptr:ptr+valid_size] = features[:valid_size]
            queue_pseudo[ptr:ptr+valid_size] = labels[:valid_size]
            self.queue_entry_batch[ptr:ptr+valid_size] = current_batch[:valid_size]
            queue_ptr += valid_size
            
            if ptr + valid_size == self.queue_size:
                self._update_smoothed_counts(queue_pseudo.sum(dim=0).clone().detach().cpu())
        elif start_direct_enqueue:
            queue_ptr = queue_ptr % self.queue_size
            ptr = int(queue_ptr)
            valid_size = min(features.size(0), self.queue_size - ptr)
            
            queue[ptr:ptr+valid_size] = features[:valid_size]
            queue_pseudo[ptr:ptr+valid_size] = labels[:valid_size]
            self.queue_entry_batch[ptr:ptr+valid_size] = current_batch[:valid_size]
            queue_ptr += valid_size
            
            if ptr + valid_size == self.queue_size:
                self._update_smoothed_counts(queue_pseudo.sum(dim=0).clone().detach().cpu())
        else:
            # print("4, 队列已满，进入缓冲区")
            self._add_to_buffer(features, labels, current_batch)
            # print('shape2', len(self.buffer['keys']), self.buffer_max)
            if len(self.buffer['keys']) >= self.buffer_max:
                queue_ptr, queue, queue_pseudo = self._maintain_queue(queue_ptr, queue, queue_pseudo)

        return queue_ptr, queue, queue_pseudo

    def _maintain_queue(self, queue_ptr, queue, queue_pseudo):
        """增强稳定性的队列维护"""
        # 合并数据
        # valid_mask = self.queue_entry_batch != -1
        # print(len(queue[valid_mask]), len(self.buffer['keys']))
        combined_keys = torch.cat([queue, self.buffer['keys']])
        combined_labels = torch.cat([queue_pseudo, self.buffer['labels']])
        combined_entry = torch.cat([self.queue_entry_batch, self.buffer['entry_batch']])
        
        # 计算当前分布
        current_counts = combined_labels.sum(dim=0)
        current_dist = current_counts.float() / current_counts.sum()
        self._check_stability(current_dist)
        
        # 计算评分并选择
        # print(combined_labels.shape, combined_entry.shape)
        scores = self._compute_eviction_scores(combined_labels.clone().detach().cpu(), combined_entry)
        # print(scores.shape,self.queue_size)
        k = self.queue_size
        
        if self.in_stable_mode:
            # 阶段1：保证每个类别最小配额
            min_per_class = max(1, int(self.queue_size * self.min_ratio))
            class_indices = []
            for c in range(self.num_classes):
                mask = combined_labels.clone().detach().cpu()[:,c] > 0
                if mask.sum() == 0:
                    continue  # 跳过无样本的类别
                entries = combined_entry[mask]
                # 保留最新样本，但不超过类别存在数
                actual_keep = min(min_per_class, len(entries))
                _, sorted_idx = torch.sort(entries, descending=True)
                class_indices.extend(mask.nonzero()[sorted_idx[:actual_keep]].tolist())

            # 阶段2：去重并优先保留必要样本
            expanded = [item for sublist in class_indices for item in sublist]
            protected_indices = list(dict.fromkeys(expanded))  # 保持顺序去重
            protected_indices = protected_indices[:self.queue_size]  # 防溢出
            
            # 阶段3：补充新鲜样本至满队列
            remaining = self.queue_size - len(protected_indices)
            if remaining > 0:
                # 按时间衰减分排序未保护样本
                all_indices = torch.arange(len(combined_keys))
                unused_mask = ~torch.isin(all_indices, torch.tensor(protected_indices))
                time_scores = 1.0 / (1.0 + self.decay_base * 
                                (self.global_batch_counter - combined_entry[unused_mask]).float())
                _, top_indices = torch.topk(time_scores, min(remaining, len(time_scores)))
                supplement = all_indices[unused_mask][top_indices].tolist()
                protected_indices += supplement
            
            # 最终保障机制：若仍不足则循环填充
            if len(protected_indices) < self.queue_size:
                cycle_indices = protected_indices * (self.queue_size // len(protected_indices) + 1)
                protected_indices = cycle_indices[:self.queue_size]
            
            keep_indices = torch.tensor(protected_indices[:self.queue_size])
            # print(len(protected_indices), len(candidate_indices), remaining, len(final_indices))
        else:
            # 常规选择
            # print(scores.shape, combined_keys.shape,k)
            _, keep_indices = torch.topk(scores, k)
        
        # 更新队列状态
        self.queue_entry_batch = combined_entry[keep_indices].clone()
        keep_indices = keep_indices.to(queue.device)
        queue = combined_keys[keep_indices].clone()
        queue_pseudo = combined_labels[keep_indices].clone()
        
        # 更新统计量
        new_counts = queue_pseudo.sum(dim=0)
        self._update_smoothed_counts(new_counts.clone().detach().cpu())
        self.label_counts = new_counts
        
        # 清空缓冲区
        self.buffer_entry_batch = self.buffer_entry_batch[0:0]
        self.buffer_keys = self.buffer_keys[0:0]
        self.buffer_labels = self.buffer_labels[0:0]
        # print('short analysis:', self.analyze_distribution(queue_pseudo).round(2))
        return queue_ptr, queue, queue_pseudo

    def _update_smoothed_counts(self, new_counts):
        """更新平滑后的类别统计量"""
        if self.smoothed_counts is None:
            self.smoothed_counts = new_counts.float()
        else:
            self.smoothed_counts = (
                self.smooth_factor * self.smoothed_counts 
                + (1 - self.smooth_factor) * new_counts.float()
            )

    def _check_stability(self, current_dist):
        """动态平衡状态检测"""
        # 计算分布差异
        diff = torch.norm(current_dist.clone().detach().cpu() - self.target_ratio, p=1).item()
        
        # 更新稳定计数器
        if diff < self.balance_thresh:
            self.stable_counter += 1
        else:
            self.stable_counter = max(0, self.stable_counter-2)
        
        # 状态切换判断
        if not self.in_stable_mode and self.stable_counter >= self.stable_window:
            # print("enter stable maintain mode")
            self.in_stable_mode = True
        elif self.in_stable_mode and self.stable_counter < self.stable_window//2:
            # print("exit stable maintain mode") 
            self.in_stable_mode = False

    def _compute_eviction_scores(self, labels, entry_batch):
        """动态评分模式切换"""
        # 存活期计算（稳定模式调整衰减曲线）
        live_span = self.global_batch_counter - entry_batch
        if self.in_stable_mode:
            # 稳定模式下加速旧样本衰减
            batch_decay = 1.0 / (1.0 + self.decay_base * live_span.float())
            return batch_decay  # 仅保留时间衰减项
        else:
            # 常规模式计算
            batch_decay = torch.where(
                live_span <= 10,
                1.0,
                1.0 / (1.0 + self.decay_base * (live_span.float() - 10))
            )
            
            # 正常计算稀缺性和冗余得分
            merged_counts = self.smoothed_counts.clone()
            merged_total = merged_counts.sum().clamp(min=1e-6)
            merged_ratio = merged_counts / merged_total
            safe_ratio = torch.clamp(merged_ratio, self.min_ratio, None)
            
            scarcity = (self.target_ratio - safe_ratio)
            scarcity_score = (labels * scarcity.unsqueeze(0)).sum(dim=1)
            
            target_count = self.target_ratio * merged_total
            overfilled = (merged_counts > 1.5 * target_count).float()
            redundancy_penalty = (labels * overfilled.unsqueeze(0)).sum(dim=1)
            
            return batch_decay * (scarcity_score - self.alpha * redundancy_penalty)

    def _add_to_buffer(self, keys, labels, entry_batch):
        """缓冲区管理"""
        
        self.buffer_keys = torch.cat([self.buffer['keys'], keys])
        self.buffer_labels = torch.cat([self.buffer['labels'], labels])
        self.buffer_entry_batch = torch.cat([
            self.buffer['entry_batch'], 
            entry_batch.long()
        ])
        # print('after add', len(self.buffer['keys']))
        # 缓冲区循环覆盖
        if len(self.buffer['keys']) > self.buffer_max:
            excess = len(self.buffer['keys']) - self.buffer_max
            self.buffer_keys = self.buffer['keys'][excess:]
            self.buffer_labels = self.buffer['labels'][excess:]
            self.buffer_entry_batch = self.buffer['entry_batch'][excess:]
        # print('after exclude', len(self.buffer['keys']))

    def analyze_distribution(self, queue_pseudo):
        """带保护的分布分析"""
        counts = queue_pseudo.sum(dim=0)
        total = counts.sum().clamp(min=1e-6)
        return (counts / total).cpu().numpy()
        # return counts.numpy()