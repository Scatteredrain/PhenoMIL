# from types import _ReturnT_co
from numpy.lib import extract
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from .randaugment import RandomAugment
import random
from PIL import Image
import os
import numpy as np
import yaml
import torch.distributed as dist
from .custom_transforms import *
from tqdm import tqdm

# from .utils_algo import generate_uniform_cv_candidate_labels
from .utils_endoscopy import get_class_map, read_json, read_paths_and_labels, LabelEnhancer

import math
from torch.utils.data import Sampler, DistributedSampler

# label enhancer    
or_groups = [[0,1],[2,3,4,5,6,7]] # pos features, neg features
and_groups = [[3,4]] # high correlated features
label_enhancer = LabelEnhancer(or_groups=or_groups, and_groups=and_groups)

class BalancedEpochSampler(DistributedSampler):
    """
    动态平衡采样器：每个epoch选取与标注样本等量的未标注样本
    Args:
        dataset: 需包含image_labels属性的数据集
        num_replicas: 分布式进程数（自动获取）
        rank: 当前进程编号（自动获取）
        shuffle: 是否打乱顺序
        seed: 随机种子
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, 
                        shuffle=shuffle, seed=seed)
        print('#[using BalancedEpochSampler]#')
        # 分离索引
        self.labeled_idx = np.where(dataset.image_labels != -1)[0].tolist()
        self.unlabeled_idx = np.where(dataset.image_labels == -1)[0].tolist()
        self.num_labeled = len(self.labeled_idx)
        
        # 验证数据集有效性
        if self.num_labeled == 0:
            raise ValueError("数据集中未找到标注样本！")
        if len(self.unlabeled_idx) == 0:
            raise ValueError("数据集中未找到未标注样本！")
        
        self.epoch = 0
        self.total_size = 2 * self.num_labeled  # 每个epoch总样本量

    def __iter__(self):
        # 生成确定性随机序列
        random.seed(self.epoch)
        np.random.seed(self.epoch)
        
        # 动态选择未标注样本
        selected_unlabeled = random.sample(self.unlabeled_idx, self.num_labeled)
        
        # 合并索引并打乱
        combined = self.labeled_idx + selected_unlabeled
        random.shuffle(combined)
        
        # 分布式分片
        indices_per_replica = self.total_size // self.num_replicas
        start_idx = self.rank * indices_per_replica
        end_idx = start_idx + indices_per_replica
        
        return iter(combined[start_idx:end_idx])

    def __len__(self):
        return self.total_size // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch
        super().set_epoch(epoch)

class NaturalDistributedSampler(DistributedSampler):
    """
    自然分布采样器：每个batch的标注比例自动匹配数据集整体比例
    Args:
        dataset: 需包含image_labels属性的数据集
        batch_size: 全局批量大小
        num_replicas: 进程数（自动获取）
        rank: 当前进程编号（自动获取）
        shuffle: 是否打乱顺序
        seed: 随机种子
    """
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        print('#[using NaturalDistributedSampler]#')
        # 计算全局标注比例
        self.labeled_idx = np.where(dataset.image_labels != -1)[0].tolist()
        self.unlabeled_idx = np.where(dataset.image_labels == -1)[0].tolist()
        self.total_samples = len(dataset)
        self.labeled_ratio = len(self.labeled_idx) / self.total_samples
        
        # 分布式参数
        self.global_batch_size = batch_size
        self.local_batch_size = math.ceil(batch_size / num_replicas)
        
        # 分层分片
        self.labeled_subsets = self._split_stratified(self.labeled_idx)
        self.unlabeled_subsets = self._split_stratified(self.unlabeled_idx)
        
        self.epoch = 0

    def _split_stratified(self, indices):
        """保持原分布的分片方法"""
        num_replicas = self.num_replicas
        per_replica = math.ceil(len(indices) / num_replicas)
        padded = indices.copy()
        while len(padded) < per_replica * num_replicas:
            padded += indices[:per_replica*num_replicas-len(padded)]
        return [padded[i*per_replica:(i+1)*per_replica] for i in range(num_replicas)]

    def __iter__(self):
        # 确定随机种子（保证分布式一致性）
        random.seed(self.epoch)
        np.random.seed(self.epoch)
        
        # 打乱各进程的子集
        local_labeled = random.sample(self.labeled_subsets[self.rank], len(self.labeled_subsets[self.rank]))
        local_unlabeled = random.sample(self.unlabeled_subsets[self.rank], len(self.unlabeled_subsets[self.rank]))
        
        # 合并索引（保持原比例）
        merged = []
        labeled_ptr = unlabeled_ptr = 0
        while labeled_ptr < len(local_labeled) or unlabeled_ptr < len(local_unlabeled):
            # 按比例采样
            expected_labeled = int(self.local_batch_size * self.labeled_ratio)
            actual_labeled = min(expected_labeled, len(local_labeled)-labeled_ptr)
            actual_unlabeled = self.local_batch_size - actual_labeled
            
            batch = (
                local_labeled[labeled_ptr : labeled_ptr+actual_labeled] +
                local_unlabeled[unlabeled_ptr : unlabeled_ptr+actual_unlabeled]
            )
            merged += batch
            
            labeled_ptr += actual_labeled
            unlabeled_ptr += actual_unlabeled
        
        return iter(merged)

    def set_epoch(self, epoch):
        self.epoch = epoch
        super().set_epoch(epoch)

    def __len__(self):
        # return math.ceil(len(self.labeled_subsets[self.rank] + self.unlabeled_subsets[self.rank]) / self.local_batch_size)
        return math.ceil(len(self.labeled_subsets[self.rank] + self.unlabeled_subsets[self.rank]))


class DistributedStratifiedSampler(DistributedSampler):
    """
    修复后的分布式分层采样器，支持精确控制标注比例
    Args:
        dataset: 目标数据集，需包含image_labels属性
        batch_size: 全局批量大小 (所有进程总和) 
        labeled_ratio: 标注样本比例 (0.0~1.0)
        num_replicas: 分布式进程数 (通常自动获取)
        rank: 当前进程ID (通常自动获取)
        shuffle: 是否打乱顺序
        seed: 随机种子
    """
    def __init__(self, dataset, batch_size, labeled_ratio=0.5, 
                 num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, 
                        shuffle=shuffle, seed=seed)
        
        print('#[using DistributedStratifiedSampler]#')
        if not 0 <= labeled_ratio <= 1:
            raise ValueError("labeled_ratio必须在[0,1]之间")
        
        # 关键修复：接收并保存batch_size参数
        self.global_batch_size = batch_size
        self.labeled_ratio = labeled_ratio
        
        # 分离标注/非标注索引
        self.labeled_indices = [i for i, lbl in enumerate(dataset.image_labels) if lbl != -1]
        self.unlabeled_indices = [i for i, lbl in enumerate(dataset.image_labels) if lbl == -1]
        
        # 计算每个进程的本地batch_size
        self.local_batch_size = math.ceil(self.global_batch_size / self.num_replicas)
        
        # 分片索引
        self.labeled_subsets = self._split_indices(self.labeled_indices)
        self.unlabeled_subsets = self._split_indices(self.unlabeled_indices)
        
        self.epoch = 0

    def _split_indices(self, indices):
        """将索引均分到各进程"""
        num_samples = len(indices)
        num_per_replica = math.ceil(num_samples / self.num_replicas)
        padded_size = num_per_replica * self.num_replicas
        indices += indices[:(padded_size - num_samples)]  # 填充重复样本
        return [indices[i*num_per_replica:(i+1)*num_per_replica] for i in range(self.num_replicas)]

    def __iter__(self):
        # 生成确定性的随机顺序
        random.seed(self.epoch)
        
        # 当前进程的索引子集
        labeled = random.sample(self.labeled_subsets[self.rank], len(self.labeled_subsets[self.rank]))
        unlabeled = random.sample(self.unlabeled_subsets[self.rank], len(self.unlabeled_subsets[self.rank]))
        
        # 计算每个本地batch的标注样本数
        per_local_labeled = max(1, int(self.local_batch_size * self.labeled_ratio))
        
        # 生成混合索引
        mixed = []
        len_labeled = len(labeled)
        len_unlabeled = len(unlabeled)
        
        total_batches = (len_labeled + len_unlabeled) // self.local_batch_size
        for _ in range(total_batches):
            # 选取标注样本
            curr_labeled = labeled[:per_local_labeled]
            labeled = labeled[per_local_labeled:]
            
            # 补充非标注样本
            need_unlabeled = self.local_batch_size - len(curr_labeled)
            curr_unlabeled = unlabeled[:need_unlabeled]
            unlabeled = unlabeled[need_unlabeled:]
            
            mixed += curr_labeled + curr_unlabeled
        
        return iter(mixed)

    def set_epoch(self, epoch):
        self.epoch = epoch
        super().set_epoch(epoch)



def generate_sequence(start: int, end: int, length: int) -> list:
    """
    生成指定范围的随机序列，满足以下规则：
    1. 当长度 ≤ 范围宽度时：元素全不重复
    2. 当长度 ＞ 范围宽度时：先产生全排列，再补充随机元素
    
    :param start: 起始值（包含）
    :param end: 结束值（包含）
    :param length: 需要生成的序列长度
    :return: 满足要求的随机序列
    """
    # 处理无效输入情况
    if length < 0:
        raise ValueError("序列长度不能为负数")
    if start > end:
        raise ValueError("起始值不能大于结束值")
    
    # 计算范围宽度
    width = end - start + 1
    numbers = list(range(start, end + 1))
    
    # 生成基础随机排列
    random.shuffle(numbers)
    
    # 构造结果序列
    if length <= width:
        return np.array(numbers[:length])
    else:
        # 全排列部分 + 补充随机部分
        remaining = length - width
        additional = [random.choice(numbers) for _ in range(remaining)]
        return np.array(numbers + additional)

def load_config(file_path='config.yaml'):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_partial_labels_WSI_cancer(img_corresponding_patient_labels):
    partialY_train = []
    for i in range(len(img_corresponding_patient_labels)):
        if img_corresponding_patient_labels[i] == 0:
            partialY_train[i] = torch.tensor([1, 0])
        else:
            partialY_train[i] = torch.tensor([0, 1])
    partialY_train = np.array(partialY_train)
    print('Average candidate num: ', partialY_train.sum(1).mean())

def get_partial_labels_HP_infection(img_labels, num_class=2):
    # clean partial label: true label need to be among the candidate labels!
    # semi-supervised learning can be similar to partial label learning
    # -> labeled data rate vs partial rate

    # HP infection here is a semi-supervised task, for we have partial annotated img label
    # WSI cancer can be seen as a semi-supervised task, cause the label in negative bag must be negative, which is true label

    partialY = torch.zeros((len(img_labels), num_class))
    for i in range(len(img_labels)):
        if img_labels[i] == 0:
            partialY[i] = torch.tensor([1, 0]) if num_class == 2 else torch.tensor([1,0,0])
        elif img_labels[i] == 1:
            partialY[i] = torch.tensor([0, 1]) if num_class == 2 else torch.tensor([0,1,0])
        elif img_labels[i] == -1:
            partialY[i] = torch.tensor([1, 1]) if num_class == 2 else torch.tensor([1,1,1])

    # print('Average candidate num: ', partialY.sum(1).mean())

    return partialY

def load_train_img_dataset(args, batch_size, debug=False, transform_path='./configs/transform.yaml'):
    print('transforms path:', transform_path)
    transform_list = load_config(transform_path)
    transform_list_weak, transform_list_strong, transform_list_test = transform_list['weak'], transform_list['strong'], transform_list['test']
    ## train with instances
    print('\n#### load the train dateset ####')
    train_dataset = PolypDataset_instances(
                    num_class = args.num_class,
                    index_root = args.train_file,
                    transform_list_weak=transform_list_weak,
                    transform_list_strong=transform_list_strong,
                    transform_list_test=transform_list_test,
                    is_transform=True, size=transform_list_weak['resize']['size'][0],debug=debug,
                    return_bag= False, is_train=True, only_labeled=args.train_only_labeled, cls_task=args.cls_task,
                    constrain_no_either_nums=args.train_constrain_no_neither_nums,
                    )#no_both=args.train_no_both, no_neither=args.train_no_neither)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    # train_sampler = DistributedStratifiedSampler(train_dataset, batch_size=batch_size, labeled_ratio=0.5, num_replicas=args.world_size, rank=dist.get_rank(), shuffle=True)
    train_sampler = NaturalDistributedSampler(train_dataset, batch_size=batch_size, num_replicas=args.world_size, rank=dist.get_rank(), shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=train_sampler is None,
                                    sampler=train_sampler,
                                    num_workers=4,
                                    pin_memory=True,
                                    # prefetch_factor=2, 
                                    drop_last=True)
  
    partialY_train = train_dataset.img_label_partialY.clone()
    return train_loader, partialY_train, train_sampler, train_dataset

def load_train_bag_dataset(args, batch_size, debug=False, transform_path='./configs/transform.yaml'):
    transform_list = load_config(transform_path)
    transform_list_weak, transform_list_strong, transform_list_test = transform_list['weak'], transform_list['strong'], transform_list['test']
    print('\n#### load the train bag-dataset... ####')
    ## train with bag
    train_dataset_bag = PolypDataset_instances(
                    index_root = args.train_bag_file,
                    transform_list_weak=transform_list_weak,
                    transform_list_strong=transform_list_strong,
                    transform_list_test=transform_list_test,
                    is_transform=True, size=transform_list_weak['resize']['size'][0],debug=debug,
                    no_both=False, no_neither=False,
                    return_bag=True, is_train=True)
    train_sampler_bag = torch.utils.data.distributed.DistributedSampler(train_dataset_bag, shuffle=True)
    # train_sampler_bag = None
    train_loader_bag = torch.utils.data.DataLoader(dataset=train_dataset_bag,
                                    batch_size=1,
                                    shuffle=train_sampler_bag is None,
                                    sampler=train_sampler_bag,
                                    num_workers=4,
                                    pin_memory=True,
                                    # prefetch_factor=2, 
                                    drop_last=True)
    return train_loader_bag, train_sampler_bag, train_dataset_bag

def load_eval_bag_dataset(args, batch_size, debug=False, transform_path='./configs/transform.yaml'):
    transform_list = load_config(transform_path)
    transform_list_weak, transform_list_strong, transform_list_test = transform_list['weak'], transform_list['strong'], transform_list['test']
    print('\n#### load the eval bag-dateset... ####')
    eval_dataset_bag = PolypDataset_instances(
                    index_root = args.test_file,
                    transform_list_weak=transform_list_weak,
                    transform_list_strong=transform_list_strong,
                    transform_list_test=transform_list_test,
                    is_transform=True, size=transform_list_weak['resize']['size'][0],debug=debug,
                    no_both=False, no_neither=False,
                    return_bag=True, is_train=False)
    # eval_sampler_bag = torch.utils.data.distributed.DistributedSampler(eval_dataset_bag, shuffle=False)
    eval_sampler_bag = None
    eval_loader_bag = torch.utils.data.DataLoader(dataset=eval_dataset_bag,
                                    batch_size=1,
                                    shuffle = False,
                                    # shuffle=eval_sampler_bag is None,
                                    # sampler=eval_sampler_bag,
                                    num_workers=4,
                                    pin_memory=True,
                                    # prefetch_factor=2, 
                                    drop_last=False)
    
    return eval_loader_bag, eval_sampler_bag, eval_dataset_bag
     
def load_eval_img_dataset(args, batch_size, debug=False, transform_path='./configs/transform.yaml'):
    transform_list = load_config(transform_path)
    transform_list_weak, transform_list_strong, transform_list_test = transform_list['weak'], transform_list['strong'], transform_list['test']
    print('\n#### load the eval dateset ###')
    eval_dataset_img = PolypDataset_instances(
                    num_class = args.num_class,
                    index_root = args.test_file ,
                    transform_list_weak=transform_list_weak,
                    transform_list_strong=transform_list_strong,
                    transform_list_test=transform_list_test,
                    is_transform=False, size=transform_list_weak['resize']['size'][0],
                    return_bag= False, is_train=False, only_labeled=args.train_only_labeled, cls_task=args.cls_task,
                    no_both=False, no_neither=True, constrain_no_either_nums=0 #args.eval_constrain_no_neither_nums
                    )#no_both=args.train_no_both, no_neither=args.train_no_neither)
    eval_loader_img = torch.utils.data.DataLoader(dataset=eval_dataset_img,
                                    batch_size=1,
                                    # shuffle = False,
                                    shuffle=False,
                                    num_workers=4,
                                    pin_memory=True,
                                    # prefetch_factor=2, 
                                    drop_last=False)

    return eval_loader_img, eval_dataset_img

def load_endoscopy(args, batch_size, debug=False, transform_path='./configs/transform.yaml'):
    print('transforms path:', transform_path)
    transform_list = load_config(transform_path)
    transform_list_weak, transform_list_strong, transform_list_test = transform_list['weak'], transform_list['strong'], transform_list['test']
    ## train with instances
    print('\n### load the train dateset ###')
    train_dataset = PolypDataset_instances(
                    num_class = args.num_class,
                    index_root = args.train_file,
                    transform_list_weak=transform_list_weak,
                    transform_list_strong=transform_list_strong,
                    transform_list_test=transform_list_test,
                    is_transform=True, size=transform_list_weak['resize']['size'][0],debug=debug,
                    return_bag= False, is_train=True, only_labeled=args.train_only_labeled, cls_task=args.cls_task,
                    constrain_no_either_nums=args.train_constrain_no_neither_nums,
                    )#no_both=args.train_no_both, no_neither=args.train_no_neither)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    # train_sampler = DistributedStratifiedSampler(train_dataset, batch_size=batch_size, labeled_ratio=0.5, num_replicas=args.world_size, rank=dist.get_rank(), shuffle=True)
    # train_sampler = NaturalDistributedSampler(train_dataset, batch_size=batch_size, num_replicas=args.world_size, rank=dist.get_rank(), shuffle=True)
    train_sampler = BalancedEpochSampler(train_dataset, num_replicas=args.world_size, rank=dist.get_rank(), shuffle=True) if args.semi_stage2 \
        else torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=train_sampler is None,
                                    sampler=train_sampler,
                                    num_workers=8,
                                    pin_memory=True,
                                    # prefetch_factor=2, 
                                    persistent_workers=True,
                                    drop_last=True)
  
    partialY_train = train_dataset.img_label_partialY.clone()
    print('\n### load the train bag-dataset... ###')
    ## train with bag
    train_dataset_bag = PolypDataset_bagsv2(
                    index_root = args.train_bag_file,
                    transform_list_weak=transform_list_weak,
                    transform_list_strong=transform_list_strong,
                    transform_list_test=transform_list_test,
                    is_transform=True, size=transform_list_weak['resize']['size'][0],debug=debug,
                    no_both=False, no_neither=False,
                    return_bag=True, is_train=True)
    train_sampler_bag = torch.utils.data.distributed.DistributedSampler(train_dataset_bag, shuffle=True)
    # train_sampler_bag = None
    train_loader_bag = torch.utils.data.DataLoader(dataset=train_dataset_bag,
                                    batch_size=24,
                                    shuffle=train_sampler_bag is None,
                                    sampler=train_sampler_bag,
                                    num_workers=8,
                                    pin_memory=True,
                                    prefetch_factor=2, 
                                    persistent_workers=True,
                                    drop_last=True)
                            
    ## eval with train
    print('\n#### load the eval bag-dateset... ####')
    eval_dataset_bag = PolypDataset_bagsv2(
                    index_root = args.test_file,
                    transform_list_weak=transform_list_weak,
                    transform_list_strong=transform_list_strong,
                    transform_list_test=transform_list_test,
                    is_transform=True, size=transform_list_weak['resize']['size'][0],debug=debug,
                    no_both=False, no_neither=False,
                    return_bag=True, is_train=False)
    # eval_sampler_bag = torch.utils.data.distributed.DistributedSampler(eval_dataset_bag, shuffle=False)
    eval_sampler_bag = None
    eval_loader_bag = torch.utils.data.DataLoader(dataset=eval_dataset_bag,
                                    batch_size=1,
                                    shuffle = False,
                                    # shuffle=eval_sampler_bag is None,
                                    # sampler=eval_sampler_bag,
                                    num_workers=8,
                                    pin_memory=True,
                                    # prefetch_factor=2, 
                                    drop_last=False)
    print('\n### load the eval img dateset: ###')
    eval_dataset_img = PolypDataset_instances(
                    num_class = args.num_class,
                    index_root = args.test_file ,
                    debug=debug,
                    transform_list_weak=transform_list_weak,
                    transform_list_strong=transform_list_strong,
                    transform_list_test=transform_list_test,
                    is_transform=False, size=transform_list_weak['resize']['size'][0],
                    return_bag= False, is_train=False, only_labeled=args.train_only_labeled, cls_task=args.cls_task,
                    no_both=False, #no_neither=False, constrain_no_either_nums=args.eval_constrain_no_neither_nums
                    no_neither=True, constrain_no_either_nums=0
                    )#no_both=args.train_no_both, no_neither=args.train_no_neither)
    eval_sampler_img = torch.utils.data.distributed.DistributedSampler(eval_dataset_img, shuffle=False)
    eval_loader_img = torch.utils.data.DataLoader(dataset=eval_dataset_img,
                                    batch_size=batch_size,
                                    # shuffle = False,
                                    sampler = eval_sampler_img,
                                    shuffle=False,
                                    num_workers=8,
                                    pin_memory=True,
                                    # prefetch_factor=2, 
                                    drop_last=False)

    return train_loader, partialY_train, train_sampler, train_dataset, \
        eval_loader_bag, eval_sampler_bag, eval_dataset_bag, \
            train_loader_bag, train_sampler_bag, train_dataset_bag, \
            eval_loader_img, eval_dataset_img

class PolypDataset_bags(Dataset):
    def __init__(self, index_root, transform_list_weak, transform_list_strong, transform_list_test, is_transform=False, is_train=True, size=352, 
                    debug=False, return_bag=False, only_labeled=False, num_class=2, 
                    eight_class=True, cls_task='multi_cls', no_both=False, no_neither=True, 
                    constrain_no_either_nums=0, add_extra_data=False, 
                    only_unlabeled=False, unlabeled_pseudo_labels=None, mask_unlabeled_valid=None):
        file_index, labels = read_paths_and_labels(index_root)
        self.is_train = is_train
        self.num_class = num_class
        # self.is_transform = is_transform
        self.return_bag = return_bag
        self.patient_label_map = read_json()
        self.only_labeled = only_labeled
        self.only_unlabeled = only_unlabeled
        self.eight_class = eight_class
        self.cls_task = cls_task
        self.no_neighter_num = 0
        self.weak_transform = self.get_transform(transform_list_weak)
        self.strong_transform = self.get_transform(transform_list_strong)
        self.test_transform = self.get_transform(transform_list_test) 

        fuji_index = []
        if debug:
            file_index = file_index[:20] + file_index[-20:]
            labels = labels[:20] + labels[-20:]
        self.bag_labels = labels

        self.is_train = is_train
        self.bags = []
        self.labels= []
        self.names = []

        self.bags_n = []
        self.labels_n= []
        self.bags_p = []
        self.labels_p = []
        self.names_n = []
        self.names_p = []

        for i, folder_path in enumerate(file_index):
            folder = folder_path.split('/')[-1]
            label = labels[i]
            if folder in fuji_index:
                continue
            images = []
            for f in os.listdir(folder_path):
                if '报告' in f:
                    continue
                # if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.BMP') or f.endswith('.JPG'):
                if f.endswith('.jpg') or f.endswith('.Jpg'):
                    images.append(os.path.join(folder_path, f))
            self.bags.append(images)
            self.names.append(folder)
            if label == 1:
                self.names_p.append(folder)
                self.labels.append(1)
                self.bags_p.append(images)
                self.labels_p.append(1)
            else:
                self.names_n.append(folder)
                self.labels.append(0)
                self.bags_n.append(images)
                self.labels_n.append(0)

    @staticmethod
    def get_transform(transform_list):
        tfs = []
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return transforms.Compose(tfs)

    def __getitem__(self, index):
        out = {'index': index}
    
        paths_one_bag = self.bags[index]
        label_bag = self.labels[index]
        bag_images = []
        for img_path in paths_one_bag:
            image = Image.open(img_path).convert('RGB')
            if self.is_train:
                sample = {"image": image}
                sample = self.weak_transform(sample)
                bag_images.append(sample["image"])
            else:
                sample = {"image": image}
                sample = self.test_transform(sample)
                bag_images.append(sample["image"])
        # print('len', len(bag_images))
        out['bag'] = bag_images
        out['labels_instance'] = [label_bag]*len(bag_images)
        out['label_bag'] = label_bag
        out['image_path'] = paths_one_bag[0]

        return out
    
    def random_choose_nocancer_img(self):
        combined = list(zip(self.bags_n, self.labels_n, self.names_n))
        random.shuffle(combined)
        self.bags_n[:], self.labels_n[:], self.names_n[:] = zip(*combined)

        # 正样本数量
        num_p = len(self.bags_p[:])

        self.bags = self.bags_p + self.bags_n[:num_p]
        self.labels = self.labels_p + self.labels_n[:num_p]
        self.names = self.names_p + self.names_n[:num_p]
        print(f'\n Balance sampling bag train dataset - aligned length: {num_p}*2')
        # combined = list(zip(self.bags, self.labels, self.names))
        # random.shuffle(combined)
        # self.bags[:], self.labels[:], self.names[:] = zip(*combined)

    def __len__(self):
        if self.is_train and len(self.bags_n) > len(self.bags_p):
            return len(self.bags_p) * 2
        else:
            return len(self.bags)

class PolypDataset_bagsv2(Dataset):
    def __init__(self, index_root, transform_list_weak, transform_list_strong, transform_list_test, is_transform=False, is_train=True, size=352, 
                    debug=False, return_bag=False, only_labeled=False, num_class=2, 
                    eight_class=True, cls_task='multi_cls', no_both=False, no_neither=True, 
                    constrain_no_either_nums=0, add_extra_data=False, 
                    only_unlabeled=False, unlabeled_pseudo_labels=None, mask_unlabeled_valid=None,
                    bag_size=7):
        file_index, labels = read_paths_and_labels(index_root)
        self.is_train = is_train
        self.num_class = num_class
        self.bag_size = bag_size
        # self.is_transform = is_transform
        self.return_bag = return_bag
        self.patient_label_map = read_json()
        self.only_labeled = only_labeled
        self.only_unlabeled = only_unlabeled
        self.eight_class = eight_class
        self.cls_task = cls_task
        self.no_neighter_num = 0
        fuji_index = []
        if debug:
            file_index = file_index[:10] + file_index[-10:]
            labels = labels[:10] + labels[-10:]
        self.bag_labels_ori = labels
        
        n_confused = 0
        idx_unlabeled = -1
        self.idxs_unlabeled = []
        
        self.bags_ori, self.bag_labels_img_ori, self.bag_labels_img_8class_ori = [], [], []

        for i, folder_path in enumerate(file_index):
            folder = folder_path.split('/')[-1]
            bag = []
            labels_img = []
            labels_img_8class = []
            
            if folder in fuji_index:
                continue
            for f in os.listdir(folder_path):
                if '报告' in f:
                    continue
                # if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.BMP') or f.endswith('.JPG'):
                if f.endswith('.jpg') or f.endswith('.Jpg'):
                    img_path = os.path.join(folder_path, f)
                    # get img label: 0 means negative, 1 means positive, -1 means unknown, -2 means confused
                    if img_path.split('/')[-4] == 'HP_image_train':
                        image_name = img_path.split('/')[-1]
                        patient_name = image_name.split('_')[0]
                        if self.eight_class:
                            img_label, image_label_8class = get_class_map(self.patient_label_map, patient_name, image_name, eight_class=True)
                        else:
                            img_label = get_class_map(self.patient_label_map, patient_name, image_name, eight_class=False)
                    else:
                        idx_unlabeled += 1
                        self.idxs_unlabeled.append(idx_unlabeled)
                        if unlabeled_pseudo_labels is not None and mask_unlabeled_valid is not None:
                            mask_valid = mask_unlabeled_valid[idx_unlabeled]
                            img_label = -1
                            image_label_8class = unlabeled_pseudo_labels[idx_unlabeled]

                        else:
                            img_label = -1
                            image_label_8class = [0.5]*8

                    # img_label [0: positive, 1: negative, -1: unknown, -2: both/confused, -3: all zero]
                    if img_label == -3 and no_neither: 
                        if self.no_neighter_num < constrain_no_either_nums:
                            # 随机数选择此次要不要
                            if random.random() < 0.5:
                                self.no_neighter_num += 1
                            else:
                                continue
                        else:
                            continue
                    if img_label == -2 and no_both:
                        n_confused += 1
                        continue
                    if self.only_labeled and img_label == -1:
                        continue
                    if self.only_unlabeled and img_label != -1:
                        continue

                    bag.append(img_path)
                    labels_img.append(img_label)
                    labels_img_8class.append(image_label_8class)

            # print('[filtered] confused data num: ', n_confused)
            '''labels_img: [0: positive, 1: negative, -1: unlabeled, -2: both/confused, -3: all zero]
                labels_img_8class: [0: unexit, 1: exit, 0.5: unlabeled]*8
            '''
            self.bags_ori.append(bag)
            self.bag_labels_img_ori.append(labels_img)
            self.bag_labels_img_8class_ori.append(labels_img_8class)
        
        self.bag_labels_ori = np.array(self.bag_labels_ori)
        self.bags, self.bag_labels, self.bag_labels_img, self.bag_labels_img_8class = self.bags_ori, self.bag_labels_ori, self.bag_labels_img_ori, self.bag_labels_img_8class_ori
        
        self.weak_transform = self.get_transform(transform_list_weak)
        self.strong_transform = self.get_transform(transform_list_strong)
        self.test_transform = self.get_transform(transform_list_test) 


    def __getitem__(self, index):
        out = {}
        paths_one_bag = self.bags[index]
        labels_one_bag = self.bag_labels[index]
        labels_image_one_bag = self.bag_labels_img[index]

        # random select self.bag_size from paths_one_bag and labels_image_one_bag
        combined = list(zip(paths_one_bag, labels_image_one_bag))
        combined = random.sample(combined, min(len(combined), self.bag_size))
        paths_one_bag, labels_image_one_bag = zip(*combined)

        bag_images = []
        for i, img_path in enumerate(paths_one_bag):
            # label_img = labels_image_one_bag[i]
            image = Image.open(img_path).convert('RGB')
            if self.is_train:
                sample = {"image": image}
                sample = self.weak_transform(sample)
                bag_images.append(sample["image"])
            else:
                sample = {"image": image}
                sample = self.test_transform(sample)
                bag_images.append(sample["image"])
        # print('len', len(bag_images))
        out['bag'] = bag_images
        out['label_bag'] = labels_one_bag
        out['labels_instance'] = labels_image_one_bag
        out['image_path'] = paths_one_bag[0]

        return out

    @staticmethod
    def get_transform(transform_list):
        tfs = []
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return transforms.Compose(tfs)
    
    def __len__(self):
        positive_idxs = np.where(self.bag_labels_ori == 1)[0]
        negative_idxs = np.where(self.bag_labels_ori == 0)[0]
        num = 2*min(len(positive_idxs), len(negative_idxs))
        return num

    def balance_label_sample(self, balance_type=1):
        # copy the less label data
        # assert self.only_labeled and (not self.return_bag)
        # assert self.true_label_num > 0
        # first relate to self.image_labels to split the positive and negative,
        # zip self.image_paths, self.image_corresponding_patient_labels, self.image_labels, self.image_corresponding_patient_indexs, self.image_labels_8class
        positive_idxs = np.where(self.bag_labels_ori == 1)
        negative_idxs = np.where(self.bag_labels_ori == 0)
        print('ori positive data num: ', len(positive_idxs[0]))
        print('ori negative data num: ', len(negative_idxs[0]))
        if balance_type == 0:
            if len(positive_idxs[0]) > len(negative_idxs[0]):
                random_idx = generate_sequence(0, len(positive_idxs[0])-1, len(negative_idxs[0]))
                positive_idxs = positive_idxs[0][random_idx]
                negative_idxs = negative_idxs[0]
            else:
                random_idx = generate_sequence(0, len(negative_idxs[0])-1, len(positive_idxs[0]))
                negative_idxs = negative_idxs[0][random_idx]
                positive_idxs = positive_idxs[0]
            
        elif balance_type == 1:
            # copy the less label data to the same length of the more label data
            if len(positive_idxs[0]) > len(negative_idxs[0]):
                random_idx_append = generate_sequence(0, len(negative_idxs[0])-1, len(positive_idxs[0]))
                negative_idxs = negative_idxs[0][random_idx_append]
                positive_idxs = positive_idxs[0]
            else:
                random_idx_append = generate_sequence(0, len(positive_idxs[0])-1, len(negative_idxs[0]))
                positive_idxs = positive_idxs[0][random_idx_append]
                negative_idxs = negative_idxs[0]
        extract_idxs = np.concatenate([positive_idxs, negative_idxs])
        np.random.shuffle(extract_idxs)
        # print('positive: ', positive_idxs)
        # print('negative: ', negative_idxs)

        # self.bags, self.bag_labels, self.bag_labels_img, self.bag_labels_img_8class = self.bags_ori[extract_idxs], self.bag_labels_ori[extract_idxs], self.bag_labels_img_ori[extract_idxs], self.bag_labels_img_8class_ori[extract_idxs]
        self.bag_labels = self.bag_labels_ori[extract_idxs]
        self.bags = [self.bags_ori[i] for i in extract_idxs]
        self.bag_labels_img = [self.bag_labels_img_ori[i] for i in extract_idxs]
        self.bag_labels_img_8class = [self.bag_labels_img_8class_ori[i] for i in extract_idxs]
        self.positive_num = len(np.where(self.bag_labels == 1)[0])
        self.negative_num = len(np.where(self.bag_labels == 0)[0])
        print('balanced positive data num: ', self.positive_num)
        print('balanced negative data num: ', self.negative_num)
        # combined = list(zip(self.bags, self.labels, self.names))
        # random.shuffle(combined)
        # self.bags[:], self.labels[:], self.names[:] = zip(*combined)

class PolypDataset_instances(Dataset):
    def __init__(self, index_root, transform_list_weak, transform_list_strong, transform_list_test, is_transform=False, is_train=True, size=352, 
                    debug=False, return_bag=False, only_labeled=False, num_class=2, 
                    eight_class=True, cls_task='multi_cls', no_both=False, no_neither=True, 
                    constrain_no_either_nums=0, add_extra_data=False, 
                    only_unlabeled=False, unlabeled_pseudo_labels=None, mask_unlabeled_valid=None):
        file_index, labels = read_paths_and_labels(index_root)
        self.is_train = is_train
        self.num_class = num_class
        # self.is_transform = is_transform
        self.return_bag = return_bag
        self.patient_label_map = read_json()
        self.only_labeled = only_labeled
        self.only_unlabeled = only_unlabeled
        self.eight_class = eight_class
        self.cls_task = cls_task
        self.no_neighter_num = 0
        fuji_index = []
        if debug:
            file_index = file_index[:10] + file_index[-10:]
            labels = labels[:10] + labels[-10:]
        self.bag_labels_ori = labels
        
        n_confused = 0
        idx_unlabeled = -1
        self.image_paths, self.image_corresponding_patient_labels, self.image_labels, self.image_corresponding_patient_indexs = [], [], [], []
        self.image_labels_8class = []
        self.idxs_unlabeled = []
        for i, folder_path in enumerate(file_index):
            folder = folder_path.split('/')[-1]
            label = self.bag_labels_ori[i]
            
            if folder in fuji_index:
                continue
            for f in os.listdir(folder_path):
                if '报告' in f:
                    continue
                # if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.BMP') or f.endswith('.JPG'):
                if f.endswith('.jpg') or f.endswith('.Jpg'):
                    img_path = os.path.join(folder_path, f)
                    # get img label: 0 means negative, 1 means positive, -1 means unknown, -2 means confused
                    if img_path.split('/')[-4] == 'HP_image_train':
                        image_name = img_path.split('/')[-1]
                        patient_name = image_name.split('_')[0]
                        if self.eight_class:
                            img_label, image_label_8class = get_class_map(self.patient_label_map, patient_name, image_name, eight_class=True)
                        else:
                            img_label = get_class_map(self.patient_label_map, patient_name, image_name, eight_class=False)
                    else:
                        idx_unlabeled += 1
                        self.idxs_unlabeled.append(idx_unlabeled)
                        if unlabeled_pseudo_labels is not None and mask_unlabeled_valid is not None:
                            mask_valid = mask_unlabeled_valid[idx_unlabeled]
                            img_label = -1
                            image_label_8class = unlabeled_pseudo_labels[idx_unlabeled]

                        else:
                            img_label = -1
                            image_label_8class = [0.5]*8

                    # img_label [0: positive, 1: negative, -1: unknown, -2: both/confused, -3: all zero]
                    if img_label == -3 and no_neither: 
                        if self.no_neighter_num < constrain_no_either_nums:
                            # 随机数选择此次要不要
                            if random.random() < 0.5:
                                self.no_neighter_num += 1
                            else:
                                continue
                        else:
                            continue
                    if img_label == -2 and no_both:
                        n_confused += 1
                        continue
                    if self.only_labeled and img_label == -1:
                        continue
                    if self.only_unlabeled and img_label != -1:
                        continue
                    self.image_paths.append(img_path)
                    self.image_corresponding_patient_labels.append(label)
                    self.image_corresponding_patient_indexs.append(i)
                    self.image_labels.append(img_label)
                    self.image_labels_8class.append(image_label_8class)
        

        if add_extra_data:
            
            file_path_jiejie = '/mnt/data/yizhenyu/data/HP识别/workspace/PiCO/configs/multi_center_training2/结节_append_paths.txt'
            with open(file_path_jiejie, 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
            self.image_paths.extend(lines)
            self.image_corresponding_patient_labels.extend([-1]*len(lines))
            self.image_corresponding_patient_indexs.extend([-1]*len(lines))
            self.image_labels.extend([1]*len(lines))
            self.image_labels_8class.extend([[0,0,0,0,0,1,0,0]]*len(lines))

            file_path_wdxxr = '/mnt/data/yizhenyu/data/HP识别/workspace/PiCO/configs/multi_center_training2/胃底腺息肉_append_paths.txt'
            with open(file_path_wdxxr, 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
            self.image_paths.extend(lines)
            self.image_corresponding_patient_labels.extend([-1]*len(lines))
            self.image_corresponding_patient_indexs.extend([-1]*len(lines))
            self.image_labels.extend([0]*len(lines))
            self.image_labels_8class.extend([[0,1,0,0,0,0,0,0]]*len(lines))



        # print('[filtered] confused data num: ', n_confused)
        '''image_labels: [0: positive, 1: negative, -1: unlabeled, -2: both/confused, -3: all zero]
            image_labels_8class: [0: unexit, 1: exit, 0.5: unlabeled]*8
        '''
        self.image_labels = np.array(self.image_labels)  
        self.image_paths, self.image_corresponding_patient_indexs, self.image_corresponding_patient_labels, self.image_labels_8class =\
            np.array(self.image_paths), np.array(self.image_corresponding_patient_indexs), np.array(self.image_corresponding_patient_labels), np.array(self.image_labels_8class)
        self.true_label_num = len(self.image_labels[self.image_labels != -1])
        self.bag_labels_ori = np.array(self.bag_labels_ori)
        print('len of valid patient',len(np.unique(self.image_corresponding_patient_indexs)))
        self.bag_labels = self.bag_labels_ori[np.unique(self.image_corresponding_patient_indexs)]
        if not self.return_bag:
            if self.cls_task == 'multi_cls':
                self.img_label_partialY = get_partial_labels_HP_infection(self.image_labels, num_class=num_class)
            elif self.cls_task == 'multi_label':
                self.img_label_partialY = torch.tensor(self.image_labels_8class)
                
            # mean number of labels per sample in self.img_label_partialY
            if not self.only_unlabeled:
                print('Within Labeled data - Average candidate num: ', self.img_label_partialY[self.image_labels != -1].sum(1).mean(dtype=torch.float32))
                print('Within Labeled data - labeled num of each class: ', self.img_label_partialY[self.image_labels != -1].sum(0))
            self.labeled_freq = self.img_label_partialY[self.image_labels != -1].sum(0) / self.true_label_num
            print('labeled data num: ', self.true_label_num)
            print('unlabeled data num: ', len(self.image_labels) - self.true_label_num)
        self.weak_transform = self.get_transform(transform_list_weak)
        self.strong_transform = self.get_transform(transform_list_strong)
        self.test_transform = self.get_transform(transform_list_test) 
        self.positive_num = len(np.where(self.image_labels == 1)[0])
        self.negative_num = len(np.where(self.image_labels == 0)[0])

        

    def __getitem__(self, index):
        out = {'index': index}
        if not self.return_bag:
            img_path = self.image_paths[index]
            img_label_true = self.image_labels[index] #[1], belong to {0,1,-1,-2,-3}
            img_label = self.img_label_partialY[index] #[C], one-hot of image_label_true
            img_label_true_8class = self.image_labels_8class[index] #[8], one-hot of 8 cls
            img_corresponding_patient_label = self.image_corresponding_patient_labels[index] # [1], belong to {0,1}
            out['label'], out['label_true'], out['label_corresponding_patient'], out['label_true_8class'] = \
                img_label, img_label_true, img_corresponding_patient_label, img_label_true_8class

            image = Image.open(img_path).convert('RGB')
            sample = {"image": image}
            if self.is_train:
                image_w = self.weak_transform(sample.copy())["image"]
                image_s = self.strong_transform(sample)["image"]
                out['image_w'], out['image_s'] = image_w, image_s
            else:
                image_eval = self.test_transform(sample)["image"]
                out['image_eval'] = image_eval
            if self.only_unlabeled:
                out['idx_unlabeled'] = self.idxs_unlabeled[index]
            # return image_w, image_s, img_label, img_label_true, img_corresponding_patient_label, index
        else:
            # print('index:', index, len(self.bag_labels))
            idxs_bag = self.image_corresponding_patient_indexs == index
            # print('valid num', idxs_bag.sum())
            paths_one_bag = self.image_paths[idxs_bag]
            labels_one_bag = self.image_corresponding_patient_labels[idxs_bag]
            bag_images = []
            for img_path in paths_one_bag:
                image = Image.open(img_path).convert('RGB')
                if self.is_train:
                    sample = {"image": image}
                    sample = self.weak_transform(sample)
                    bag_images.append(sample["image"])
                else:
                    sample = {"image": image}
                    sample = self.test_transform(sample)
                    bag_images.append(sample["image"])
            # print('len', len(bag_images))
            out['bag'] = bag_images
            out['labels_instance'] = labels_one_bag
            out['label_bag'] = labels_one_bag[0]
            out['image_path'] = paths_one_bag[0]

        return out

    @staticmethod
    def get_transform(transform_list):
        tfs = []
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return transforms.Compose(tfs)
    
    def __len__(self):
        if not self.return_bag:
            return len(self.image_labels)
        else:
            return len(self.bag_labels)

    def balance_label_sample(self, balance_type=1):
        # copy the less label data
        assert self.only_labeled and (not self.return_bag)
        assert self.true_label_num > 0
        # first relate to self.image_labels to split the positive and negative,
        # zip self.image_paths, self.image_corresponding_patient_labels, self.image_labels, self.image_corresponding_patient_indexs, self.image_labels_8class
        positive_idxs = np.where(self.image_labels == 1)
        negative_idxs = np.where(self.image_labels == 0)
        print('positive data num: ', len(positive_idxs[0]))
        print('negative data num: ', len(negative_idxs[0]))
        if balance_type == 0:
            if len(positive_idxs[0]) > len(negative_idxs[0]):
                random_idx = generate_sequence(0, len(positive_idxs[0])-1, len(negative_idxs[0]))
                positive_idxs = positive_idxs[0][random_idx]
                negative_idxs = negative_idxs[0]
            else:
                random_idx = generate_sequence(0, len(negative_idxs[0])-1, len(positive_idxs[0]))
                negative_idxs = negative_idxs[0][random_idx]
                positive_idxs = positive_idxs[0]
            
        elif balance_type == 1:
            # copy the less label data to the same length of the more label data
            if len(positive_idxs[0]) > len(negative_idxs[0]):
                random_idx_append = generate_sequence(0, len(negative_idxs[0])-1, len(positive_idxs[0]))
                negative_idxs = negative_idxs[0][random_idx_append]
                positive_idxs = positive_idxs[0]
            else:
                random_idx_append = generate_sequence(0, len(positive_idxs[0])-1, len(negative_idxs[0]))
                positive_idxs = positive_idxs[0][random_idx_append]
                negative_idxs = negative_idxs[0]
        extract_idxs = np.concatenate([positive_idxs, negative_idxs])
        np.random.shuffle(extract_idxs)
        # print('positive: ', positive_idxs)
        # print('negative: ', negative_idxs)

        self.image_paths, self.image_corresponding_patient_labels,self.image_labels,self.image_corresponding_patient_indexs, self.image_labels_8class = \
            self.image_paths[extract_idxs], self.image_corresponding_patient_labels[extract_idxs],self.image_labels[extract_idxs], \
            self.image_corresponding_patient_indexs[extract_idxs], self.image_labels_8class[extract_idxs]
        self.positive_num = len(np.where(self.image_labels == 1)[0])
        self.negative_num = len(np.where(self.image_labels == 0)[0])
        if not self.return_bag:
            self.img_label_partialY = get_partial_labels_HP_infection(self.image_labels, num_class=self.num_class)
        # combined = list(zip(self.bags, self.labels, self.names))
        # random.shuffle(combined)
        # self.bags[:], self.labels[:], self.names[:] = zip(*combined)



class PolypDataset(Dataset):
    def __init__(self, index_root, transform_list, is_transform, is_train=True, testsize=299, debug=False):
        self.patient_label_map = read_json('/mnt/data/yizhenyu/data/HP识别/前瞻性验证/description/2368_5.json')
        file_index, labels = read_paths_and_labels(index_root)
        if debug:
            file_index = file_index[:100] + file_index[-100:]
            labels = labels[:100] + labels[-100:]
        # if not is_train:
        #     fuji_index = read_index_from_file("/mnt/data/yizhenyu/data/HP识别/workspace/MIL-HP_yzy/configs/old_data_cross_fold/fuji.txt")
        # else:
        #     fuji_index = []
        fuji_index = []
        self.is_train = is_train
        self.bags = []
        self.labels= []
        self.names = []

        self.bags_n = []
        self.labels_n= []
        self.bags_p = []
        self.labels_p = []
        self.names_n = []
        self.names_p = []

        for i, folder_path in enumerate(file_index):
            folder = folder_path.split('/')[-1]
            label = labels[i]
            if folder in fuji_index:
                continue
            images = []
            for f in os.listdir(folder_path):
                if '报告' in f:
                    continue
                # if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.BMP') or f.endswith('.JPG'):
                if f.endswith('.jpg') or f.endswith('.Jpg'):
                    images.append(os.path.join(folder_path, f))
            self.bags.append(images)
            self.names.append(folder)
            if label == 1:
                self.names_p.append(folder)
                self.labels.append(1)
                self.bags_p.append(images)
                self.labels_p.append(1)
            else:
                self.names_n.append(folder)
                self.labels.append(0)
                self.bags_n.append(images)
                self.labels_n.append(0)
        # print('positive data num: ', len(self.bags_p))
        # print('negative data num: ', len(self.bags_n))
        if is_transform:
            self.transform = self.get_transform(transform_list)
        else:
            self.transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    @staticmethod
    def get_transform(transform_list):
        tfs = []
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return transforms.Compose(tfs)

    def __getitem__(self, index):
        images = self.bags[index]
        bag_label = self.labels[index]
 
        images_tensor = []
        img_labels = []
        name_tensor = []
        for img_path in images:
            image = Image.open(img_path).convert('RGB')
            if self.is_train:
                sample = {"image": image}
                sample = self.transform(sample)
                images_tensor.append(sample["image"])
            else:
                sample = self.transform(image)
                images_tensor.append(sample)
            
            # get img class label (eight-class one hot)
            img_label = get_class_map(self.patient_label_map, os.path.split(img_path)[-1].split('_')[0], os.path.split(img_path)[-1], eight_class=False)
            img_labels.append(torch.tensor(img_label))

            name_tensor.append(os.path.split(img_path)[-1])
        
        return torch.stack(images_tensor), torch.tensor(bag_label), torch.tensor(img_labels), name_tensor
    
    def random_choose_nocancer_img(self):
        combined = list(zip(self.bags_n, self.labels_n, self.names_n))
        random.shuffle(combined)
        self.bags_n[:], self.labels_n[:], self.names_n[:] = zip(*combined)

        # 正样本数量
        num_p = len(self.bags_p[:])

        self.bags = self.bags_p + self.bags_n[:num_p]
        self.labels = self.labels_p + self.labels_n[:num_p]
        self.names = self.names_p + self.names_n[:num_p]
        # combined = list(zip(self.bags, self.labels, self.names))
        # random.shuffle(combined)
        # self.bags[:], self.labels[:], self.names[:] = zip(*combined)

    def __len__(self):
        if self.is_train and len(self.bags_n) > len(self.bags_p):
            return len(self.bags_p) * 2
        else:
            return len(self.bags)

def generate_Class_Aware_thresholds(args, batch_size, model, pos_label_freq, debug=False, transform_path='./configs/transform_new.yaml'):

    ## load train dataset
    # print('transforms path:', transform_path)
    print('\n calculating class-distribution aware thresholds...')
    transform_list = load_config(transform_path)
    transform_list_weak, transform_list_strong, transform_list_test = transform_list['weak'], transform_list['strong'], transform_list['test']
    ## train with instances
    train_dataset = PolypDataset_instances(
                    num_class = args.num_class,
                    index_root = args.train_file,
                    transform_list_weak=transform_list_weak,
                    transform_list_strong=transform_list_strong,
                    transform_list_test=transform_list_test,
                    is_transform=True, size=transform_list_weak['resize']['size'][0],debug=debug,
                    return_bag= False, is_train=False, cls_task=args.cls_task,
                    constrain_no_either_nums=args.train_constrain_no_neither_nums,
                    only_unlabeled=True if not debug else False, only_labeled=False
                    )#no_both=args.train_no_both, no_neither=args.train_no_neither)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size*4,
                                    shuffle=train_sampler is None,
                                    sampler=train_sampler,
                                    num_workers=4,
                                    pin_memory=True,
                                    # prefetch_factor=2, 
                                    drop_last=True)
  
    ## eval to get distribution on train dataset
    outputs_cls, outputs_prot = [], []
    orders = []
    model.eval()
    with torch.no_grad():
        for i, pack in enumerate(train_loader):
            X = pack['image_eval']
            orders.append(pack['idx_unlabeled'].cpu().numpy())
            score_clsHead, _, _, score_prot = model(X, eval_only=True)
            outputs_cls.append(score_clsHead.squeeze(0).detach()) # [B,C]
            outputs_prot.append(score_prot.squeeze(0).detach())
    outputs_cls = torch.cat(outputs_cls, dim=0)
    outputs_prot = torch.cat(outputs_prot, dim=0)

    dist.barrier()
    world_size = dist.get_world_size()

    all_outputs_cls = [torch.empty_like(outputs_cls) for _ in range(world_size)]
    dist.all_gather(all_outputs_cls, outputs_cls)
    all_outputs_cls = torch.cat(all_outputs_cls, dim=0)

    all_outputs_prot = [torch.empty_like(outputs_prot) for _ in range(world_size)]
    dist.all_gather(all_outputs_prot, outputs_prot)
    all_outputs_prot = torch.cat(all_outputs_prot, dim=0)

    ## get binary thresholds
    n_ub, c = all_outputs_cls.shape
    indices = [int(x)-1 for x in pos_label_freq*n_ub]

    sorted_outputs_cls = torch.sort(all_outputs_cls, dim=0, descending=True)[0]
    thre_vec_cls = sorted_outputs_cls[indices, range(c)]

    sorted_outputs_prot =torch.sort(all_outputs_prot, dim=0, descending=True)[0]
    thre_vec_prot = sorted_outputs_prot[indices, range(c)]

    print('labeled data class freq: ', pos_label_freq)
    print('gained thresholds of cls head:', thre_vec_cls)
    print('gained thresholds of prot head:', thre_vec_prot)

    return thre_vec_cls.cpu(), thre_vec_prot.cpu()

def generate_pseudolabel_dataset(args, batch_size, model, pos_label_freq, debug=False, transform_path='./configs/transform.yaml'):

    ## load train dataset
    print('transforms path:', transform_path)
    transform_list = load_config(transform_path)
    transform_list_weak, transform_list_strong, transform_list_test = transform_list['weak'], transform_list['strong'], transform_list['test']
    ## train with instances
    train_dataset = PolypDataset_instances(
                    num_class = args.num_class,
                    index_root = args.train_file,
                    transform_list_weak=transform_list_weak,
                    transform_list_strong=transform_list_strong,
                    transform_list_test=transform_list_test,
                    is_transform=True, size=transform_list_weak['resize']['size'][0],debug=debug,
                    return_bag= False, is_train=False, cls_task=args.cls_task,
                    constrain_no_either_nums=args.train_constrain_no_neither_nums,
                    only_unlabeled=True, only_labeled=False
                    )#no_both=args.train_no_both, no_neither=args.train_no_neither)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=train_sampler is None,
                                    sampler=train_sampler,
                                    num_workers=4,
                                    pin_memory=True,
                                    # prefetch_factor=2, 
                                    drop_last=True)
  
    ## eval to get distribution on train dataset
    outputs_cls, outputs_prot = [], []
    orders = []
    model.eval()
    with torch.no_grad():
        for i, pack in enumerate(train_loader):
            if i % (len(train_loader)//10) == 0 and i != 0: print(i, end=' ')
            X = pack['image_eval']
            orders.append(pack['idx_unlabeled'].cpu().numpy())
            score_clsHead, q, feat_encoder, score_prot = model(X, eval_only=True)
            outputs_cls.append(score_clsHead.squeeze(0).detach()) # [B,C]
            outputs_prot.append(q.squeeze(0).detach())
    outputs_cls = torch.cat(outputs_cls, dim=0)
    outputs_prot = torch.cat(outputs_prot, dim=0)

    dist.barrier()
    world_size = dist.get_world_size()
    all_outputs = [torch.empty_like(outputs) for _ in range(world_size)]
    # 收集所有GPU的更新区域和值
    dist.all_gather(all_outputs, outputs)
    outputs = np.concatenate(outputs.cpu().numpy())

    ## get pseudo-label
    sorted_outputs = -np.sort(-outputs, axis=0)
    n_ub = len(outputs)
    indices = [int(x)-1 for x in pos_label_freq*n_ub]
    thre_vec = sorted_outputs[indices, range(outputs.shape[1])]
    pseudo_labels = (outputs>=thre_vec).astype(np.float32)

    mask = np.zeros_like(pseudo_labels)
    pos_indices = [int(x)-1 for x in args.pos_per*pos_label_freq*n_ub]
    pos_thre = sorted_outputs[pos_indices, range(outputs.shape[1])]
    mask[outputs>=pos_thre]=1

    neg_indices = [int(x)-1 for x in args.neg_per*(1-pos_label_freq)*n_ub]
    sorted_outputs_neg = np.sort(outputs, axis=0)
    neg_thre = sorted_outputs_neg[neg_indices, range(outputs.shape[1])]
    mask[outputs<=neg_thre]=1
    
    # orders is a list between 0 and n_ub, sort the pseudo_labels and mask according to descending order
    orders = np.concatenate(orders)
    orders = orders.argsort()
    pseudo_labels = pseudo_labels[orders]
    mask = mask[orders]

    ## reload pseudo-labeled unlabeled-train-dataset
    train_dataset = PolypDataset_instances(
                    num_class = args.num_class,
                    index_root = args.train_file,
                    transform_list_weak=transform_list_weak,
                    transform_list_strong=transform_list_strong,
                    transform_list_test=transform_list_test,
                    is_transform=True, size=transform_list_weak['resize']['size'][0],debug=debug,
                    return_bag= False, is_train=False, only_labeled=args.train_only_labeled, cls_task=args.cls_task,
                    constrain_no_either_nums=args.train_constrain_no_neither_nums,
                    only_unlabeled=False, unlabeled_pseudo_labels=pseudo_labels, mask_unlabeled_valid=mask
                    )#no_both=args.train_no_both, no_neither=args.train_no_neither)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=train_sampler is None,
                                    sampler=train_sampler,
                                    num_workers=4,
                                    pin_memory=True,
                                    # prefetch_factor=2, 
                                    drop_last=True)
    partialY_train = train_dataset.img_label_partialY.clone()

    return train_loader, partialY_train, train_sampler, train_dataset


def load_polyp_seg_dataset(args, transform_path='./configs/transform.yaml'):

    transform_list = load_config(transform_path)
    transform_list_weak, transform_list_strong, transform_list_test = transform_list['weak'], transform_list['strong'], transform_list['test']

    image_root = '/mnt/data/yizhenyu/data_extra/dataset/TrainDataset/images/'
    gt_root = image_root.replace('images','masks')
    train_dataset = PolypDataset_seg(image_root, gt_root, augmentations=True, transform_list_strong=transform_list_strong, transform_list_test=transform_list_test, debug=args.debug)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=64,
                                    shuffle=train_sampler is None,
                                    sampler=train_sampler,
                                    num_workers=4,
                                    pin_memory=True,
                                    # prefetch_factor=2, 
                                    drop_last=True)

    test_dataset = PolypDataset_seg(image_root.replace('TrainDataset','TestDataset/test'), gt_root.replace('TrainDataset','TestDataset/test'), augmentations=False, transform_list_strong=transform_list_strong, transform_list_test=transform_list_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=4,
                                    pin_memory=True,
                                    # prefetch_factor=2, 
                                    )

    return train_loader, test_loader


class PolypDataset_seg(Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, augmentations, transform_list_strong, transform_list_test, debug=False):
        self.augmentations = augmentations

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if debug:
            self.images = self.images[:10]
            self.gts = self.gts[:10]
        self.filter_files()
        self.size = len(self.images)
        if self.augmentations and transform_list_strong is not None:
            print('Using transform.yaml')
            self.transform = self.get_transform(transform_list_strong)
            
        elif transform_list_test is not None:
            print('no augmentation')
            self.transform = self.get_transform(transform_list_test)
            
    @staticmethod
    def get_transform(transform_list):
        tfs = []
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return transforms.Compose(tfs)

    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7

        sample = {'image': image, 'gt': gt}
        sample = self.transform(sample)
        image = sample['image']
        gt = sample['gt']

        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


