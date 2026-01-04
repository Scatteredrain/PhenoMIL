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
from .custom_transforms import *
# from .utils_algo import generate_uniform_cv_candidate_labels
from .utils_endoscopy import get_class_map, read_json, read_paths_and_labels

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



def load_endoscopy(args, batch_size, debug=False, transform_path='./configs/transform.yaml'):
    print('transforms path:', transform_path)
    transform_list = load_config(transform_path)
    transform_list_weak, transform_list_strong, transform_list_test = transform_list['weak'], transform_list['strong'], transform_list['test']
    ## train with instances
    train_dataset = PolypDataset_instances(
                    num_class = args.num_class,
                    index_root = args.train_file ,
                    transform_list_weak=transform_list_weak,
                    transform_list_strong=transform_list_strong,
                    transform_list_test=transform_list_test,
                    is_transform=True, size=transform_list_weak['resize']['size'][0],debug=debug,
                    return_bag= False, is_train=True, only_labeled=args.train_only_labeled, cls_task=args.cls_task,
                    constrain_no_either_nums=args.train_constrain_no_neither_nums,
                    )#no_both=args.train_no_both, no_neither=args.train_no_neither)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=train_sampler is None,
                                    sampler=train_sampler,
                                    num_workers=4,
                                    pin_memory=True,
                                    # prefetch_factor=2, 
                                    drop_last=True)
  
    partialY_train = train_dataset.img_label_partialY.clone()
    ## train with bag
    train_dataset_bag = PolypDataset_instances(
                    index_root = args.train_file,
                    transform_list_weak=transform_list_weak,
                    transform_list_strong=transform_list_strong,
                    transform_list_test=transform_list_test,
                    is_transform=True, size=transform_list_weak['resize']['size'][0],debug=debug,
                    return_bag=True, is_train=True)
    train_sampler_bag = torch.utils.data.distributed.DistributedSampler(train_dataset_bag, shuffle=True)
    train_loader_bag = torch.utils.data.DataLoader(dataset=train_dataset_bag,
                                    batch_size=1,
                                    # shuffle = False,
                                    shuffle=train_sampler_bag is None,
                                    sampler=train_sampler_bag,
                                    num_workers=4,
                                    pin_memory=True,
                                    # prefetch_factor=2, 
                                    drop_last=True)
                            
    ## eval with train
    eval_dataset_bag = PolypDataset_instances(
                    index_root = args.test_file,
                    transform_list_weak=transform_list_weak,
                    transform_list_strong=transform_list_strong,
                    transform_list_test=transform_list_test,
                    is_transform=True, size=transform_list_weak['resize']['size'][0],debug=debug,
                    return_bag=True, is_train=False)
    eval_sampler_bag = torch.utils.data.distributed.DistributedSampler(eval_dataset_bag, shuffle=False)
    eval_loader_bag = torch.utils.data.DataLoader(dataset=eval_dataset_bag,
                                    batch_size=1,
                                    # shuffle = False,
                                    shuffle=eval_sampler_bag is None,
                                    sampler=eval_sampler_bag,
                                    num_workers=4,
                                    pin_memory=True,
                                    # prefetch_factor=2, 
                                    drop_last=False)

    return train_loader, partialY_train, train_sampler, train_dataset, \
        eval_loader_bag, eval_sampler_bag, eval_dataset_bag, \
            train_loader_bag, train_sampler_bag, train_dataset_bag

class PolypDataset_instances(Dataset):
    def __init__(self, index_root, transform_list_weak, transform_list_strong, transform_list_test, is_transform, is_train=True, size=352, 
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
        self.bag_labels = labels
        n_confused = 0
        self.image_paths, self.image_corresponding_patient_labels, self.image_labels, self.image_corresponding_patient_indexs = [], [], [], []
        self.image_labels_8class = []
        for i, folder_path in enumerate(file_index):
            folder = folder_path.split('/')[-1]
            label = self.bag_labels[i]
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
        if not self.return_bag:
            if self.cls_task == 'multi_cls':
                self.img_label_partialY = get_partial_labels_HP_infection(self.image_labels, num_class=num_class)
            elif self.cls_task == 'multi_label':
                self.img_label_partialY = torch.tensor(self.image_labels_8class)
                
            # mean number of labels per sample in self.img_label_partialY
            print('Labeled data - Average candidate num: ', self.img_label_partialY[self.image_labels != -1].sum(1).mean(dtype=torch.float32))
            print('Labeled data - labeled num of each class: ', self.img_label_partialY[self.image_labels != -1].sum(0))
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
            img_label_true = self.image_labels[index] #[1], belong to {0,1,-1}
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

            # return image_w, image_s, img_label, img_label_true, img_corresponding_patient_label, index
        else:
            idxs_bag = self.image_corresponding_patient_indexs == index
            paths_one_bag = self.image_paths[idxs_bag]
            labels_one_bag = self.image_labels[idxs_bag]
            label_bag = self.bag_labels[index]
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
                
            out['bag'] = bag_images
            out['labels_instance'] = labels_one_bag
            out['label_bag'] = label_bag
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


# def generate_pseudolabel_dataset(args, batch_size, model, debug=False, transform_path='./configs/transform.yaml'):

#     ## load train dataset
#     print('transforms path:', transform_path)
#     transform_list = load_config(transform_path)
#     transform_list_weak, transform_list_strong, transform_list_test = transform_list['weak'], transform_list['strong'], transform_list['test']
#     ## train with instances
#     train_dataset = PolypDataset_instances(
#                     num_class = args.num_class,
#                     index_root = args.train_file,
#                     transform_list_weak=transform_list_weak,
#                     transform_list_strong=transform_list_strong,
#                     transform_list_test=transform_list_test,
#                     is_transform=True, size=transform_list_weak['resize']['size'][0],debug=debug,
#                     return_bag= False, is_train=False, only_labeled=args.train_only_labeled, cls_task=args.cls_task,
#                     constrain_no_either_nums=args.train_constrain_no_neither_nums,
#                     )#no_both=args.train_no_both, no_neither=args.train_no_neither)
#     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                     batch_size=batch_size,
#                                     shuffle=train_sampler is None,
#                                     sampler=train_sampler,
#                                     num_workers=4,
#                                     pin_memory=True,
#                                     # prefetch_factor=2, 
#                                     drop_last=True)
  
                      
#     ## eval to get distribution on train dataset
#     with torch.no_grad():
#         model.eval()
#         for i, pack in enumerate(train_loader):
#             X = pack['image_eval']
#             score_clsHead, q, feat_encoder, score_prot = model(X, eval_only=True)

