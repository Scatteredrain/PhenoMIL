import os
import torch
import yaml
import cv2
import argparse

import numpy as np

from easydict import EasyDict as ed

import torch.distributed as dist
from PIL import Image, ImageDraw, ImageFont
import json
from collections import defaultdict
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score, classification_report, roc_auc_score, confusion_matrix

import torch
import torch.distributed as dist

def read_json(root = '/mnt/data/yizhenyu/data/HP识别/HP_image_train_annotations/2570_5.json'):
    # 从文件中读取 JSON 数据并解析为 Python 对象
    with open(root, 'r') as file:
        data = json.load(file)
        
    # patient_info[study_id, folder_name, img list&label, [标注者1, 任务id], [标注者2, 任务id]]
    total_info = []
    for subtask in data["subTaskList"]:
        # if subtask["status"] == 0:
        #     continue
        patient_info = []
        subtask_id = subtask["subTaskId"]
        # print(subtask_id)
        annotator = subtask["annotator"]
        study_id = subtask["fileList"][0]["studyInstanceUid"]

        if study_id not in [row[0] for row in total_info]:
            patient_info.append(study_id)

            file_path1 = subtask["fileList"][0]["sopFileList"][0]["httpUrl"]
            file_path2 = subtask["fileList"][0]["sopFileList"][0]["ossKey"]
            abs_folder_path = file_path1.split("/")
            folder_path = abs_folder_path[-3] + "/" + abs_folder_path[-2]
            patient_info.append(folder_path)
            # print(folder_path)
            image_list = []
            
            # if len(subtask["findingList"])!=1:
            #     print("Warning!", len(subtask["findingList"]), subtask["fileList"][-1]["studyInstanceUid"])

            for file in subtask["fileList"]:
                image_name_label = []
                tmp_study_id = file["studyInstanceUid"]
                tmp_file_path1 = file["sopFileList"][0]["httpUrl"]
                tmp_file_path2 = file["sopFileList"][0]["ossKey"]
                tmp_file_path = tmp_file_path2.split("/")
                image_path = tmp_file_path[-3] + "/" + tmp_file_path[-2] + "/" + tmp_file_path[-1]
                sop_id = file["sopFileList"][0]["sopInstanceUid"]
                image_name_label.append(image_path)
                image_name_label.append(sop_id)
                # if tmp_study_id != study_id or tmp_file_path1 != tmp_file_path2:
                #     print("Warning!", file["sopFileList"][0]["httpUrl"])

                if len(subtask["findingList"]) != 1:
                    labels = []
                else:
                    labels = subtask["findingList"][0]["annotation"]["sopAnnotationData"]

                label_list = []
                for label in labels:
                    label_sop_id = label["sopInstanceUid"]
                    if label_sop_id == sop_id:
                        for tmp_label in label["data"]:
                            if "condition" in tmp_label["tag"] and tmp_label["tag"]["condition"] != "":
                                label_class = tmp_label["tag"]["condition"]
                            else:
                                label_class = "Unknown"
                            label_list.append([tmp_label["coord"], label_class])
                    else:
                        continue
                # label_list.append(annotator)
                image_name_label.append(label_list)
                # if len(image_name_label) > 3 or len(image_name_label) <= 1:
                #     print("Warning! len(image_name_label) == 1")
                while len(image_name_label) < 3:
                    image_name_label.append([])
                # if len(image_name_label) == 2:
                #     image_name_label.append([])
                # elif len(image_name_label) != 3:
                #     print("Warning! len(image_name_label) == 1")
                image_list.append(image_name_label)
            patient_info.append(image_list)

            subTaskId = subtask["subTaskId"]
            patient_info.append([annotator, subtask_id])
            total_info.append(patient_info)
        else:
            patient_index = [row[0] for row in total_info].index(study_id)
            patient_info = total_info[patient_index]
            patient_info.append([annotator, subtask_id])
            image_list = patient_info[2]
            for file in subtask["fileList"]:
                tmp_file_path1 = file["sopFileList"][0]["httpUrl"]
                tmp_file_path2 = file["sopFileList"][0]["ossKey"]
                tmp_file_path = tmp_file_path2.split("/")
                image_path = tmp_file_path[-3] + "/" + tmp_file_path[-2] + "/" + tmp_file_path[-1]
                sop_id = file["sopFileList"][0]["sopInstanceUid"]
                file_index = [row[0] for row in image_list].index(image_path)
                image_name_label = image_list[file_index]

                labels = subtask["findingList"][0]["annotation"]["sopAnnotationData"]
                label_list = []
                for label in labels:
                    label_sop_id = label["sopInstanceUid"]
                    if label_sop_id == sop_id:
                        for tmp_label in label["data"]:
                            if "condition" in tmp_label["tag"] and tmp_label["tag"]["condition"] != "":
                                label_class = tmp_label["tag"]["condition"]
                            else:
                                label_class = "Unknown"
                            # if sop_id == "3e4cec00-987a-4ce7-a746-d58847de1bfe":
                            #     print(label_class)
                            label_list.append([tmp_label["coord"], label_class])
                    else:
                        continue
                # label_list.append(annotator)
                image_name_label.append(label_list)
                # if len(image_name_label) > 4 or len(image_name_label) <= 2:
                #     print("Warning! len(image_name_label)")
                while len(image_name_label) < 4:
                    image_name_label.append([])

                # if len(image_name_label) == 3:
                #     image_name_label.append([])
                # elif len(image_name_label) != 4:
                #     print("Warning! len(image_name_label) == 1")
    # total_info.pop()

    patient_label_map = {}
    for patient in total_info:
        patient_index = patient[1].split("/")[-1]
        # if patient_index == "42":
        #     print(patient)
        patient_img = {}
        for img in patient[2]:
            img_name = img[0].split("/")[-1]
            doctor1 = patient[3][0]
            # doctor2 = patient[4][0]
            set1 = set()
            for label1 in img[2]:
                for split_label in label1[-1].split(","):
                    set1.add(split_label.strip())
            set2 = set()
            # for label2 in img[3]:
            #     set2.add(label2[-1])
            # label1 = {doctor1: set1, doctor2: set2}
            label1 = {doctor1: set1}
            patient_img[img_name] = label1
        patient_label_map[patient_index] = patient_img
    return patient_label_map



def calculate_metrics(y_true, y_pred, y_proba=None):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    TN = ((y_pred == 0) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    
    metrics = {
        'Acc': (TP + TN) / (TP + TN + FP + FN),
        'Sen': TP / (TP + FN) if (TP + FN) > 0 else 0,
        'Pre': TP / (TP + FP) if (TP + FP) > 0 else 0,
        'Spe': TN / (TN + FP) if (TN + FP) > 0 else 0,
        'NPV': TN / (TN + FN) if (TN + FN) > 0 else 0,
        'P': int(TP + FN),
        'N': int(TN + FP),
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
    }
    
    if y_proba is not None:
        from sklearn.metrics import roc_auc_score
        metrics['AUC'] = roc_auc_score(y_true, y_proba)
        # f1-score
        metrics['F1'] = f1_score(y_true, y_pred, average='binary')
    
    return {k: round(v, 4) if isinstance(v, float) else v 
            for k, v in metrics.items()}

def confidence_interval(y_true, y_pred, n_samples=1000, n_jobs=-1, metric = roc_auc_score, alpha=100-95):
    def inner():
        y_true_res, y_pred_res = resample(y_true, y_pred, stratify=y_true)
        return metric(y_true_res, y_pred_res)

    bootstrap_metrics = Parallel(n_jobs=n_jobs)(delayed(inner)() for _ in range(n_samples))
    lower_ci = np.percentile(bootstrap_metrics, alpha/2)
    upper_ci = np.percentile(bootstrap_metrics, 100 - alpha/2)
    return lower_ci, upper_ci


def print_result(prefix, dict):
    acc = dict["correct"] / (dict["total"]) * 100 if dict["total"] > 0 else 0
    recall = dict["TP"] / (dict["TP"] + dict["FN"]) * 100 if (dict["TP"] + dict["FN"]) > 0 else 0
    pre = dict["TP"] / (dict["TP"] + dict["FP"]) * 100 if (dict["TP"] + dict["FP"]) > 0 else 0
    npv = dict["TN"] / (dict["TN"] + dict["FN"]) * 100 if (dict["TN"] + dict["FN"]) > 0 else 0
    spe = dict["TN"] / (dict["TN"] + dict["FP"]) * 100 if (dict["TN"] + dict["FP"]) > 0 else 0
    f1 = 2 * recall * pre / (recall + pre + 1e-6)
    auc = roc_auc_score(dict["all_labels"], dict["all_probs"]) * 100
    if len(prefix) >= 16:
        prefix += "\t"
    else:
        prefix += "\t\t"
    print(prefix, "Total:{}, Corr:{}, TP:{}, FP:{}, TN:{}, FN:{}\n\t\t\tAcc: {:.2f}, Rec: {:.2f}, Pre: {:.2f}, Spe: {:.2f}, F1: {:.2f}, Auc: {:.2f}"
          .format(dict["total"], dict["correct"], dict["TP"], dict["FP"], dict["TN"], dict["FN"],
                  acc, recall, pre, spe, f1, auc))
    return [dict["total"], dict["correct"], dict["TP"], dict["FP"], dict["TN"], dict["FN"],
            acc, recall, pre, spe, f1, auc]

def print_result_imglabel(prefix, dict):
    # 计算评价指标
    y_true = [[round(i) for i in sublist] for sublist in dict["all_labels"]]
    y_pred = [[round(i) for i in sublist] for sublist in dict["all_probs"]]
    nor_img = 0
    abnor_img = 0
    true_pred = 0
    false_pred = 0
    true_pred_nor = 0
    true_pred_abnor = 0
    false_pred_nor = 0
    false_pred_abnor = 0

    for i, item in enumerate(y_true):
        if all(x == 0 for x in item):
            nor_img+=1
        else:
            abnor_img+=1
        if y_true[i] == y_pred[i]:
            true_pred += 1
            if all(x == 0 for x in y_true[i]):
                true_pred_nor+=1
            else:
                true_pred_abnor+=1
        else:
            false_pred += 1
            if all(x == 0 for x in y_true[i]):
                false_pred_nor+=1
            else:
                false_pred_abnor+=1
            

    acc = true_pred / (nor_img + abnor_img) * 100
    sen = true_pred_abnor / abnor_img * 100
    spe = true_pred_nor / nor_img * 100

    subset_accuracy = accuracy_score(y_true, y_pred) * 100
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0) * 100
    recall_micro = recall_score(y_true, y_pred, average='micro') * 100
    f1_micro = f1_score(y_true, y_pred, average='micro') * 100
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    recall_macro = recall_score(y_true, y_pred, average='macro') * 100
    f1_macro = f1_score(y_true, y_pred, average='macro') * 100

    # 打印结果
    print("Acc: {:.4f}, Sen: {:.4f}, Spe: {:.4f}".format(acc, sen, spe))
    print("Acc: {:.4f}, Sen(Micro): {:.4f}, Pre(Micro): {:.4f}, F1 (Micro): {:.4f}, Sen(Macro): {:.4f}, Pre(Macro): {:.4f}, F1 (Macro): {:.4f}"
          .format(subset_accuracy, recall_micro, precision_micro, f1_micro, recall_macro, precision_macro, f1_macro))

    # 输出更详细的分类报告
    print("分类报告 (按标签):")
    print(classification_report(y_true, y_pred, digits=6, zero_division=0))
    return [acc, sen, spe, subset_accuracy, recall_micro, precision_micro, f1_micro, recall_macro, precision_macro, f1_macro]

def pat_level_pred(pat_dict_merged):
    hp_th = 2
    pat_label = []
    pat_pred = []
    for key, value in pat_dict_merged.items():
        pat_name = int(key.split("/")[-1])
        pat_label.append(1 if pat_name <= 570 else 0)
        pat_score = 0
        for item in value:
            pat_score += 1 if item != 3 and item != 7 else 0
        pat_pred.append(1 if pat_score >= hp_th else 0)
    accuracy = accuracy_score(pat_label, pat_pred)
    recall = recall_score(pat_label, pat_pred)
    precision = precision_score(pat_label, pat_pred)
    tn, fp, fn, tp = confusion_matrix(pat_label, pat_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return tp, fp, tn, fn, accuracy, recall, specificity

def pat_level_pred2(pat_dict_merged):
    hp_th = 3
    pat_label = []
    pat_pred = []
    for key, value in pat_dict_merged.items():
        pat_name = int(key.split("/")[-1])
        pat_label.append(1 if pat_name <= 570 else 0)
        pat_score = 0
        for item in value:
            if item == 0:
                pat_score += 2
            elif item == 1:
                pat_score += 2
            elif item == 2:
                pat_score += 2
            elif item == 3:
                pat_score += -2
            elif item == 4:
                pat_score += 3
            elif item == 5:
                pat_score += 2
            elif item == 6:
                pat_score += 2
            elif item == 7:
                pat_score += -2
            else:
                print("False class!")
            pat_score += 1 if item != 3 and item != 7 else 0
        pat_pred.append(1 if pat_score >= hp_th else 0)
    accuracy = accuracy_score(pat_label, pat_pred)
    recall = recall_score(pat_label, pat_pred)
    precision = precision_score(pat_label, pat_pred)
    tn, fp, fn, tp = confusion_matrix(pat_label, pat_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return tp, fp, tn, fn, accuracy, recall, specificity

def result_process(dict, pred, label, instance_num, all_labels, all_values):
    dict["total"] += instance_num
    dict["correct"] += (pred == label).sum().item()
    dict["TP"] += ((pred == 1) & (label == 1)).sum().item()
    dict["FP"] += ((pred == 1) & (label != 1)).sum().item()
    dict["FN"] += ((pred != 1) & (label == 1)).sum().item()
    dict["TN"] += ((pred != 1) & (label != 1)).sum().item()
    dict["all_labels"].extend(all_labels)
    dict["all_probs"].extend(all_values)
    return dict

def flag_process(patient_pred_lstm, patient_pred_img_mean, patient_pred_pat_center, 
                patient_pred_img_center_mean, labels, opt):
    if ((patient_pred_lstm == 1) & (labels == 1)):
        flag = "_TP"
    elif ((patient_pred_lstm == 1) & (labels != 1)):
        flag = "_FP"
    elif ((patient_pred_lstm != 1) & (labels == 1)):
        flag = "_FN"
    elif ((patient_pred_lstm != 1) & (labels != 1)):
        flag = "_TN"

    if ((patient_pred_img_mean == 1) & (labels == 1)):
        flag += "_TP"
    elif ((patient_pred_img_mean == 1) & (labels != 1)):
        flag += "_FP"
    elif ((patient_pred_img_mean != 1) & (labels == 1)):
        flag += "_FN"
    elif ((patient_pred_img_mean != 1) & (labels != 1)):
        flag += "_TN"
    
    if opt.is_center_loss:
        if ((patient_pred_pat_center == 1) & (labels == 1)):
            flag += "_TP"
        elif ((patient_pred_pat_center == 1) & (labels != 1)):
            flag += "_FP"
        elif ((patient_pred_pat_center != 1) & (labels == 1)):
            flag += "_FN"
        elif ((patient_pred_pat_center != 1) & (labels != 1)):
            flag += "_TN"
    
    if opt.is_img_center_loss:
        if ((patient_pred_img_center_mean == 1) & (labels == 1)):
            flag += "_TP"
        elif ((patient_pred_img_center_mean == 1) & (labels != 1)):
            flag += "_FP"
        elif ((patient_pred_img_center_mean != 1) & (labels == 1)):
            flag += "_FN"
        elif ((patient_pred_img_center_mean != 1) & (labels != 1)):
            flag += "_TN"
    
    return flag


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/MIL.yaml')
    parser.add_argument('--checkpoint_dir', type=str, default='snapshots/debug')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--MaskDrop_threshold', default=0.7, type=float, 
                    help='MaskDrop_threshold for high sims in aggregating de-redundancy')
    parser.add_argument('--finetune', type=str, default=None, help='finetune from checkpoint')
    parser.add_argument('--finetune_path', type=str, default=None, help='finetune from checkpoint')
    parser.add_argument('--pooling', type=str, default='Mean')
    parser.add_argument('--frozen', type=bool, default=False)

    parser.add_argument('--dataset', type=str, default='old')
    args = parser.parse_args()
    
    cuda_visible_devices = None
    local_rank = -1

    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        cuda_visible_devices = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]
    if "LOCAL_RANK" in os.environ.keys():
        local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank == -1:
        device_num = 1
    elif cuda_visible_devices is None:
        device_num = torch.cuda.device_count()
    else:
        device_num = len(cuda_visible_devices)

    args.device_num = device_num
    args.local_rank = local_rank

    return args

def load_config(config_dir):
    return ed(yaml.load(open(config_dir), yaml.FullLoader))

def to_cuda(sample):
    for key in sample.keys():
        if type(sample[key]) == torch.Tensor:
            sample[key] = sample[key].cuda()
    return sample

def ddp_sync(args, total_infer, correct_infer, TP_infer, FP_infer, FN_infer, TN_infer, N_infer, P_infer):
    ddp_total = torch.tensor(total_infer).to(args.local_rank)
    dist.all_reduce(ddp_total, op=dist.ReduceOp.SUM)
    ddp_total = ddp_total.item()

    ddp_corr = torch.tensor(correct_infer).to(args.local_rank)
    dist.all_reduce(ddp_corr, op=dist.ReduceOp.SUM)
    ddp_corr = ddp_corr.item()
    # print(correct_infer, ddp_corr)

    ddp_TP = torch.tensor(TP_infer).to(args.local_rank)
    dist.all_reduce(ddp_TP, op=dist.ReduceOp.SUM)
    ddp_TP = ddp_TP.item()

    ddp_FP = torch.tensor(FP_infer).to(args.local_rank)
    dist.all_reduce(ddp_FP, op=dist.ReduceOp.SUM)
    ddp_FP = ddp_FP.item()

    ddp_FN = torch.tensor(FN_infer).to(args.local_rank)
    dist.all_reduce(ddp_FN, op=dist.ReduceOp.SUM)
    ddp_FN = ddp_FN.item()

    ddp_TN = torch.tensor(TN_infer).to(args.local_rank)
    dist.all_reduce(ddp_TN, op=dist.ReduceOp.SUM)
    ddp_TN = ddp_TN.item()

    ddp_N = torch.tensor(N_infer).to(args.local_rank)
    dist.all_reduce(ddp_N, op=dist.ReduceOp.SUM)
    ddp_N = ddp_N.item()
    ddp_P = torch.tensor(P_infer).to(args.local_rank)
    dist.all_reduce(ddp_P, op=dist.ReduceOp.SUM)
    ddp_P = ddp_P.item()

    accuracy = 100 * ddp_corr / ddp_total
    recall = ddp_TP / (ddp_TP + ddp_FN) * 100 if (ddp_TP + ddp_FN) > 0 else 0
    precision = ddp_TP / (ddp_TP + ddp_FP) * 100 if (ddp_TP + ddp_FP) > 0 else 0
    spe = ddp_TN / (ddp_TN + ddp_FP) * 100 if (ddp_TN + ddp_FP) > 0 else 0

    return ddp_total, ddp_corr, ddp_TP, ddp_FP, ddp_FN, ddp_TN, ddp_N, ddp_P, accuracy, recall, precision, spe

def gather_lists(local_list):
    # 获取当前进程的全局通信组大小
    world_size = dist.get_world_size()
    
    # 创建一个用于存储所有进程数据的张量
    # 每个进程都需要和其他进程合并，因此列表大小必须是 world_size
    gathered_lists = [torch.zeros_like(local_list.clone().detach()) for _ in range(world_size)]
    
    # 为什么这里要用 torch.tensor(local_list) ?
    # 因为我们要进行 all_gather 操作，它需要和张量交互
    local_tensor = local_list.clone().detach()
    
    # 进行 all_gather 操作
    dist.all_gather(gathered_lists, local_tensor)
    
    # 合并所有收集到的列表
    merged_list = []
    for lst in gathered_lists:
        merged_list.extend(lst.tolist())  # 将每个张量转换为列表并合并

    return merged_list

def merge_dicts_across_processes(local_dict):
    # 确保所有的进程都调用了这个函数
    world_size = dist.get_world_size()
    
    # 使用all_gather_object收集所有进程的字典到一个列表中
    all_dicts = [None for _ in range(world_size)]
    dist.all_gather_object(all_dicts, local_dict)
    
    merged_dict = {}
    for key in local_dict.keys():
        # 对于数值类型的键值，进行求和
        if isinstance(local_dict[key], (int, float)):
            merged_dict[key] = sum(d[key] for d in all_dicts)
        # 对于列表，可以直接合并（这里简单地串联，具体逻辑可能根据需求调整）
        elif isinstance(local_dict[key], list):
            merged_dict[key] = [item for sublist in [d.setdefault(key, []) for d in all_dicts] for item in sublist]
        else:
            raise ValueError(f"Unsupported type for merging: {type(local_dict[key])}")
    return merged_dict

def get_class_map(input_dict):
    class_map = {"萎缩": "A", "胃底胃体斑点状发红": "SR", "弥漫性发红": "DR", 
                "RAC": "RAC", "RAC清晰": "RAC", "结节": "N", "胃底腺息肉": "FGP", "白浊粘液": "SM", 
                "皱壁增宽": "HGF", "Unknown":"UN"}
    results = []
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

    # 去重并按字母顺序排序
    unique_sorted = sorted(list(set(results)))
    return ','.join(unique_sorted)

def show_importance(bags, bags_importance_scores, names, label_info, size=(200, 200), ncol=7):
    # bags: NxKxCxHxW; importance_scores: NxK; N=1
    org = (5, 20)  # 文本左下角在图像上的位置(x, y)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    img_title_font_scale = 2 * font_scale
    # color = (255, 255, 255)
    img_title_thickness = 1
    color = (0, 255, 0)  # 绿色
    thickness = 1
    for bag, importance_scores in zip(bags, bags_importance_scores):
        assert len(names) == len(bag)
        show_imgs = []
        show_imgs_one_line = []
        idxs = torch.argsort(importance_scores, descending=True)
        for i, idx in enumerate(idxs, start=1):
            name = names[idx][0]
            log = bag[idx].clone().cpu().detach().numpy().squeeze()
            score = importance_scores[idx].clone().cpu().detach().numpy().squeeze()
            score = "{:.2f}%".format(score*100) 
            log = log.transpose(1,2,0)
            log = (log - log.min()) / (log.max() - log.min() + 1e-8)
            log *= 255
            log = log.astype(np.uint8)
            log = cv2.cvtColor(log, cv2.COLOR_BGR2RGB)
            log = cv2.resize(log, size)
            log = cv2.putText(log, score, org, font, font_scale, color, thickness, cv2.LINE_AA)

            if name != "zeros":
                patient_name = name.split('_')[0]
                if patient_name in label_info and name in label_info[patient_name]:
                    text = get_class_map(label_info[patient_name][name])
                    log = cv2.putText(log, text, (5,87), font, img_title_font_scale, color, img_title_thickness, cv2.LINE_AA)

            if i % ncol == 1:
                show_imgs_one_line = []
            show_imgs_one_line.append(log)
            if i % ncol == 0:
                show_imgs_one_line = np.hstack(show_imgs_one_line)
                show_imgs.append(show_imgs_one_line)
    return np.vstack(show_imgs)


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*decay