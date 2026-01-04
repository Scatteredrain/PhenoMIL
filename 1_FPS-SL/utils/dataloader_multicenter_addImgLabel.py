import os

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import csv
from utils.custom_transforms import *
from utils.utils import read_json
import random

def get_class_map(all_dict, patient_name, img_name, eight_class=False):
    class_map = {"萎缩": "A", "胃底胃体斑点状发红": "SR", "弥漫性发红": "DR", 
                "RAC": "RAC", "RAC清晰": "RAC", "结节": "N", 
                "胃底腺息肉": "FGP", "白浊粘液": "SM", "皱壁增宽": "HGF", 
                "Unknown":"UN"}
    
    if patient_name in all_dict and img_name in all_dict[patient_name]:
        one_hot_results = [0]*10 
        results = []
        input_dict = all_dict[patient_name][img_name]
        for key, value in input_dict.items():
            # 处理集合类型值
            if isinstance(value, set):
                if not value:  # 排除空集合
                    continue
                for item in value:
                    code = class_map.get(item)
                    if code:
                        results.append(code)
            # 处理非集合类型值（如字符串）
            else:
                code = class_map.get(value)
                if code:
                    results.append(code)

        ## results to eight-class one hot: [A, SR, DR, RAC&RAC清晰, N, FGP, SM, HGF]
        for i, key in enumerate(class_map.values()):
            if key in results:
                one_hot_results[i] = 1
        one_hot_results[3] = min(one_hot_results[3] + one_hot_results[4],1)
        one_hot_results.pop(4)
        one_hot_results.pop(-1)
        if one_hot_results[3] == 1:
            single_result = 0
        else:
            single_result = min(sum(one_hot_results)-one_hot_results[3],1)

    else:
        # not img label -> return [0, 0, 0, 0, 0, 0, 0, 0]
        one_hot_results = [0]*8
        single_result = -1

    if not eight_class:
        return single_result # 0 for negative feature, 1 for positive feature, -1 for unknown feature

    return one_hot_results
    
def read_index_from_file(file_path):
    """
    从文件读取列表，每行去除换行符。

    参数:
        file_path (str): 文件路径

    返回:
        index_list (list): 文件夹名列表
    """
    index_list = []
    with open(file_path, 'r') as file:
        for line in file:
            index_list.append(line.strip())
    return index_list

def read_paths_and_labels(filename):
    paths = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for idx, row in enumerate(reader, 1):
            if len(row) != 2:
                continue
            path, label = row
            path = path.strip()
            label = label.strip()
            if not path or not label:
                continue
            paths.append(path)
            labels.append(int(label))
    # return paths[:100] + paths[-100:], labels[:100] + labels[-100:]
    return paths, labels


class PolypDataset(data.Dataset):
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
        

class PolypDataset_test(data.Dataset):
    def __init__(self, index_root, transform_list, is_transform, is_train=True, testsize=352):
        file_index, labels = read_paths_and_labels(index_root)
        # problem_index = read_index_from_file("/nas/qingcheng.xjw/workspace/MIL-HP/configs/new_data_cross_fold/problem_patient.txt")
        self.is_train = is_train
        self.bags = []
        self.labels= []
        self.names = []

        self.bags_n = []
        self.labels_n= []
        self.names_n = []
        self.bags_p = []
        self.labels_p= []
        self.names_p = []

        for i, folder_path in enumerate(file_index):
            if 'HP_multicenter_wenfuyinew' in folder_path:
                folder_path = folder_path.replace('HP_multicenter_wenfuyinew','HP_multicenter_wenfuyinew_nonbi')
            folder = folder_path.split('/')[-4]
            label = labels[i]
            images = []
            for f in os.listdir(folder_path):
                if '报告' in f:
                    continue
                # if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.BMP') or f.endswith('.JPG'):
                if f.endswith('.jpg') or f.endswith('.Jpg'):
                    images.append(os.path.join(folder_path, f))
            
            if len(images) < 7:
                continue
            self.bags.append(images)
            self.labels.append(label)
            self.names.append(folder)
            if label == 1:
                # self.labels.append(1)
                self.bags_p.append(images)
                self.names_p.append(folder)
                self.labels_p.append(1)
            else:
                # self.labels.append(0)
                self.bags_n.append(images)
                self.names_n.append(folder)
                self.labels_n.append(0)

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
        label = self.labels[index]
        center = self.names[index]
        images_tensor = []
        label_tensor = []
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
            
            label_tensor.append(torch.tensor(label))
            name_tensor.append(os.path.split(img_path)[-1])
        
        return torch.stack(images_tensor), torch.tensor(label_tensor), name_tensor, center
    
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
        if self.is_train:
            return len(self.bags_p) * 2
        else:
            return len(self.bags)
            
    def shengzhongyi_split(self):
        ## ['湖滨','城西'or'西溪','钱塘' or '下沙']
        return None



class PolypDataset_test_wzzx(data.Dataset):
    def __init__(self, index_root, hp_image_root, nohp_image_root, transform_list, is_transform, is_train=True, testsize=352):
        file_index = read_index_from_file(index_root)
        # problem_index = read_index_from_file("/nas/qingcheng.xjw/workspace/MIL-HP/configs/new_data_cross_fold/problem_patient.txt")
        self.is_train = is_train
        self.bags = []
        self.labels= []
        self.names = []

        self.bags_n = []
        self.labels_n= []
        self.names_n = []
        self.bags_p = []
        self.labels_p= []
        self.names_p = []

        for date in os.listdir(hp_image_root):
            if date not in file_index:
                continue
            images = []
            for f in os.listdir(os.path.join(hp_image_root, date)):
                if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.BMP') or f.endswith('.JPG') or f.endswith('.Jpg'):
                # if f.endswith('.jpg'):
                    images.append(os.path.join(hp_image_root, date, f))
            
            if len(images) < 7:
                continue
            self.bags.append(images)
            self.names.append(date)
            self.labels.append(1)
            
            self.bags_p.append(images)
            self.names_p.append(date)
            self.labels_p.append(1)
        
        for date in os.listdir(nohp_image_root):
            if date not in file_index:
                continue
            images = []
            for f in os.listdir(os.path.join(nohp_image_root, date)):
                # if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.BMP') or f.endswith('.JPG') or f.endswith('.Jpg'):
                if f.endswith('.jpg'):
                    images.append(os.path.join(nohp_image_root, date, f))
            
            if len(images) < 7:
                continue
            self.bags.append(images)
            self.names.append(date)
            self.labels.append(0)

            self.bags_n.append(images)
            self.names_n.append(date)
            self.labels_n.append(0)

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
        label = self.labels[index]
        name = self.names[index]
        images_tensor = []
        label_tensor = []
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
            
            label_tensor.append(torch.tensor(label))
            name_tensor.append(os.path.split(img_path)[-1])
        
        return torch.stack(images_tensor), torch.tensor(label_tensor), name_tensor
    
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
        if self.is_train:
            return len(self.bags_p) * 2
        else:
            return len(self.bags)
    

class PolypDataset_imglabel(data.Dataset):
    def __init__(self, image_root, index_root, img_label_root, transform_list, is_transform, is_train=True, testsize=299):
        img_label_index = read_index_from_file(img_label_root)
        file_index = read_index_from_file(index_root)

        if not is_train:
            fuji_index = read_index_from_file("/nas/qingcheng.xjw/workspace/MIL-HP/configs/old_data_cross_fold/fuji.txt")
        else:
            fuji_index = []
        fuji_index = []
        self.is_train = is_train

        self.imgs = []
        self.labels = []
        self.names = []
        self.pat = set()

        self.imgs_p = []
        self.labels_p = []
        self.names_p = []
        self.pat_p = set()
        self.imgs_n = []
        self.labels_n = []
        self.names_n = []
        self.pat_n = set()

        for img in img_label_index:
            img_name = os.path.join(image_root, img.split(" ")[0])
            pat_index = img_name.split("/")[-2]
            if pat_index not in file_index:
                continue
            img_label = img.split(" ")[1:]
            self.names.append(img_name)
            self.imgs.append(img_name)
            self.labels.append([int(x) for x in img_label])
            self.pat.add(pat_index)

            if len(img_label) == 0:
                self.names_n.append(img_name)
                self.imgs_n.append(img_name)
                self.labels_n.append([int(x) for x in img_label])
                self.pat_n.add(pat_index)
            else:
                self.names_p.append(img_name)
                self.imgs_p.append(img_name)
                self.labels_p.append([int(x) for x in img_label])
                self.pat_p.add(pat_index)

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
        image_path = self.imgs[index]
        label = self.labels[index]
        label_one_hot = torch.zeros(8)
        # if len(label) == 0:
        #     label_one_hot[8] = 1
        for idx, label in enumerate(label):
            label_one_hot[label] = 1

        image = Image.open(image_path).convert('RGB')
        if self.is_train:
            sample = {"image": image}
            sample = self.transform(sample)
            image_tensor = sample["image"]
        else:
            sample = self.transform(image)
            image_tensor = sample
        
        return image_tensor, label_one_hot, image_path
    
    def random_choose_nocancer_img(self):
        combined = list(zip(self.imgs_n, self.labels_n, self.names_n))
        random.shuffle(combined)
        self.imgs_n[:], self.labels_n[:], self.names_n[:] = zip(*combined)

        # 正样本数量
        num_p = len(self.imgs_p[:])

        self.imgs = self.imgs_p + self.imgs_n[:num_p]
        self.labels = self.labels_p + self.labels_n[:num_p]
        self.names = self.names_p + self.names_n[:num_p]
        # combined = list(zip(self.bags, self.labels, self.names))
        # random.shuffle(combined)
        # self.bags[:], self.labels[:], self.names[:] = zip(*combined)

    def __len__(self):
        if self.is_train:
            return len(self.imgs_p) * 2
        else:
            return len(self.imgs)


class PolypDataset_combine(data.Dataset):
    def __init__(self, bags, labels, names, bags_n, labels_n, names_n, bags_p, labels_p, names_p, transform_list, is_transform, is_train=True, testsize=352):
        self.is_train = is_train
        self.bags = bags
        self.labels= labels
        self.names = names

        self.bags_n = bags_n
        self.labels_n= labels_n
        self.names_n = names_n
        self.bags_p = bags_p
        self.labels_p= labels_p
        self.names_p = names_p

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
        label = self.labels[index]
        name = self.names[index]
        images_tensor = []
        label_tensor = []
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
            
            label_tensor.append(torch.tensor(label))
            name_tensor.append(os.path.split(img_path)[-1])
        
        return torch.stack(images_tensor), torch.tensor(label_tensor), name_tensor
    
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
        if self.is_train:
            return len(self.bags_p) * 2
        else:
            return len(self.bags)


class PolypDataset_video(data.Dataset):
    def __init__(self, index_root, hp_image_root, nohp_image_root, transform_list, is_transform, is_train=True, testsize=352,
                 sample_time=None, sample_frame=None):
        file_index = read_index_from_file(index_root)
        pat_index = {}
        for pat in file_index:
            pat_name = pat.split(" ")[0]
            pat_index[pat_name] = pat.split(" ")[1:]
        # problem_index = read_index_from_file("/nas/qingcheng.xjw/workspace/MIL-HP/configs/new_data_cross_fold/problem_patient.txt")
        self.is_train = is_train
        self.bags = []
        self.labels= []
        self.names = []

        self.bags_n = []
        self.labels_n= []
        self.names_n = []
        self.bags_p = []
        self.labels_p= []
        self.names_p = []

        for date in os.listdir(hp_image_root):
            if date not in pat_index:
                continue
            images = []
            for f in os.listdir(os.path.join(hp_image_root, date)):
                # if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.BMP') or f.endswith('.JPG') or f.endswith('.Jpg'):
                if f.endswith('png'):
                    images.append(os.path.join(hp_image_root, date, f))
            
            fps = int(float(pat_index[date][0]))
            duration = float(pat_index[date][1])
            frame_count = int(float(pat_index[date][2]))
            if sample_time is not None:
                sample = round(18 * sample_time)
            if sample_frame is not None:
                sample = frame_count//sample_frame

            if duration < 120:
                continue
            images = images[::sample]
            self.bags.append(images)
            self.names.append(date)
            self.labels.append(1)
            
            self.bags_p.append(images)
            self.names_p.append(date)
            self.labels_p.append(1)
        
        for date in os.listdir(nohp_image_root):
            if date not in pat_index:
                continue
            images = []
            for f in os.listdir(os.path.join(nohp_image_root, date)):
                # if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.BMP') or f.endswith('.JPG') or f.endswith('.Jpg'):
                if f.endswith('png'):
                    images.append(os.path.join(nohp_image_root, date, f))
            
            fps = int(float(pat_index[date][0]))
            duration = float(pat_index[date][1])
            frame_count = int(float(pat_index[date][2]))
            if sample_time is not None:
                sample = round(18 * sample_time)
            if sample_frame is not None:
                sample = frame_count//sample_frame

            if duration < 120:
                continue
            images = images[::sample]
            self.bags.append(images)
            self.names.append(date)
            self.labels.append(0)

            self.bags_n.append(images)
            self.names_n.append(date)
            self.labels_n.append(0)

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
        label = self.labels[index]
        name = self.names[index]
        images_tensor = []
        label_tensor = []
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
            
            label_tensor.append(torch.tensor(label))
            name_tensor.append(os.path.split(img_path)[-1])
        
        return torch.stack(images_tensor), torch.tensor(label_tensor), name_tensor
    
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
        if self.is_train:
            return len(self.bags_p) * 2
        else:
            return len(self.bags)


    