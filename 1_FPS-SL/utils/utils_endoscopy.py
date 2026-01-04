import json
import csv
import numpy as np
import torch

class LabelEnhancer:
    def __init__(self, or_groups=None, and_groups=None):
        """
        初始化标签增强器
        :param or_groups: 或组合列表 (每个元素为索引列表), 如 [[0,1], [2,3]]
        :param and_groups: 与组合列表, 如 [[4,5], [6,7]]
        """
        self.or_groups = or_groups if or_groups is not None else []
        self.and_groups = and_groups if and_groups is not None else []
    
    def transform(self, X):
        """
        对输入标签矩阵进行增强
        :param X: 输入标签矩阵, 支持 numpy 或 torch.Tensor, 形状为 [B, C]
        :return: 增强后的标签矩阵, 形状为 [B, C + M + N] (M=或组合数, N=与组合数)
        """
        # 类型和维度检查
        assert X.ndim == 2, "输入必须是二维矩阵"
        input_is_tensor = isinstance(X, torch.Tensor)
        
        # 生成新列
        new_cols = []
        
        # 处理或组合 ---------------------------------------------------
        for group in self.or_groups:
            if input_is_tensor:
                or_col = torch.any(X[:, group], dim=1, keepdim=True)
            else:
                or_col = np.any(X[:, group], axis=1, keepdims=True)
            new_cols.append(or_col)
        
        # 处理与组合 ---------------------------------------------------
        for group in self.and_groups:
            if input_is_tensor:
                and_col = torch.all(X[:, group], dim=1, keepdim=True)
            else:
                and_col = np.all(X[:, group], axis=1, keepdims=True)
            new_cols.append(and_col)
        
        # 拼接新列 -----------------------------------------------------
        if new_cols:
            if input_is_tensor:
                new_cols = torch.cat(new_cols, dim=1)
                enhanced_X = torch.cat([X, new_cols], dim=1)
            else:
                new_cols = np.concatenate(new_cols, axis=1)
                enhanced_X = np.concatenate([X, new_cols], axis=1)
        else:
            enhanced_X = X.clone() if input_is_tensor else X.copy()
        
        return enhanced_X

def Label_enhance(X):
    or_groups = [[0,1], [2,3,4,5,6,7]]
    new_cols = []
    
    for group in or_groups:
        if isinstance(X, torch.Tensor):
            or_col = torch.any(X[:, group], dim=1, keepdim=True)
        else:
            or_col = np.any(X[:, group], axis=1, keepdims=True)
        new_cols.append(or_col)

def get_class_map(all_dict, patient_name, img_name, eight_class=True):
    class_map = {"RAC": "RAC", "RAC清晰": "RAC", "胃底腺息肉": "FGP", 
                "萎缩": "A", "胃底胃体斑点状发红": "SR", "弥漫性发红": "DR", 
                "结节": "N", "白浊粘液": "SM", "皱壁增宽": "HGF", 
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
        one_hot_results[1] = min(one_hot_results[0] + one_hot_results[1],1)
        one_hot_results.pop(0)
        one_hot_results.pop(-1)

        if (one_hot_results[0] + one_hot_results[1]) > 0:
            if (sum(one_hot_results)-one_hot_results[0]-one_hot_results[1]) > 0:
                # exist both positive and negative features 
                # print(results, patient_name)
                single_result = -2
            else:
                single_result = 0
        elif (sum(one_hot_results)-one_hot_results[0]-one_hot_results[1]) >= 1:
            single_result = 1
        else:
            single_result = -3
    else:
        print('not in json! so unlabeled')
        # not img label -> return [1,1,1,1,1,1,1,1]
        one_hot_results = [0.5]*8
        single_result = -1

    if eight_class:
        return single_result, one_hot_results 
    else:
      # 0 for negative feature, 1 for positive feature, -1 for unknown feature
      return single_result

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
                    # print("Warning!", file["sopFileList"][0]["httpUrl"])

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
                    # print("Warning! len(image_name_label) == 1")
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
                    # print("Warning! len(image_name_label)")
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
            # print(patient)
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

def calculate_metrics(y_true, y_pred, y_proba=None):
    # 判断是不是np.array, 不是转化为np.array
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)   

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
        if not isinstance(y_proba, np.ndarray):
            y_proba = np.array(y_proba)
        from sklearn.metrics import roc_auc_score
        metrics['AUC'] = roc_auc_score(y_true, y_proba)
    
    return {k: round(v, 4) if isinstance(v, float) else v 
            for k, v in metrics.items()}


# 在训练循环中添加梯度布局验证
def check_gradient_layout(model):
    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            if param.data.stride() != param.grad.stride():
                print(f"参数 {name} 的梯度布局不匹配:")
                print(f"数据步长: {param.data.stride()}")
                print(f"梯度步长: {param.grad.stride()}")
                # 自动修复梯度布局
                param.grad = param.grad.contiguous()
                print("已修复为连续内存布局")