import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle

from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score, average_precision_score, confusion_matrix, hamming_loss, f1_score

def calculate_multilabel(   
    labels: np.ndarray,
    outputs_cls: np.ndarray,
    outputs_prot: np.ndarray,
    threshold: float = 0.5,
) -> dict:

    micro_metrics = evaluate_multilabel_micro_auc(labels, outputs_cls, outputs_prot)
    macro_metrics_cls = calculate_multilabel_metrics(labels, (outputs_cls>0.5).astype(int), outputs_cls)
    macro_metrics_prot = calculate_multilabel_metrics(labels, (outputs_prot>0.5).astype(int), outputs_prot)

    return {
        **micro_metrics,
        'macro_cls': macro_metrics_cls,
        'macro_prot': macro_metrics_prot,
    }

    return None

def calculate_multilabel_metrics(y_true, y_pred, y_proba):
    """多标签八分类指标计算"""
    # 逐类别计算混淆矩阵
    cm = multilabel_confusion_matrix(y_true, y_pred)
    
    # 初始化存储
    macro_metrics = {'AUC': 0., 'AP': 0., 'F1': 0.}
    class_metrics = []
    
    for i in range(y_true.shape[1]):
        # 单类别指标
        tn, fp, fn, tp = cm[i].ravel()
        auc = roc_auc_score(y_true[:, i], y_proba[:, i])
        ap = average_precision_score(y_true[:, i], y_proba[:, i])
        # f1 = f1_score(y_true[:, i], y_proba[:, i], average='binary')
        
        # 更新宏平均
        macro_metrics['AUC'] += auc
        macro_metrics['AP'] += ap
        
        
        # 记录各分类结果
        class_metrics.append({
            f'Class_{i}_AUC': auc,
            f'Class_{i}_AP': ap,
            f'Class_{i}_TP': tp,
            f'Class_{i}_FP': fp,
            f'Class_{i}_FN': fn,
            f'Class_{i}_TN': tn,
            f'Class_{i}_Sen': tp / (tp + fn) if (tp + fn) > 0 else 0,
            f'Class_{i}_Spe': tn / (tn + fp) if (tn + fp) > 0 else 0,
            f'Class_{i}_Acc': (tp + tn) / (tp + tn + fp + fn)
        })
    
    # 计算宏平均
    num_classes = y_true.shape[1]
    macro_metrics = {k: v/num_classes for k, v in macro_metrics.items()}
    macro_metrics['F1'] = f1_score(y_true, y_pred, average='macro')
    macro_metrics['class_wise'] = class_metrics    # 合并结果
    # return {**macro_metrics, **class_metrics}
    return macro_metrics


def evaluate_multilabel_micro_auc(
    labels: np.ndarray,
    outputs_cls: np.ndarray,
    outputs_prot: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    评估多标签分类任务的两个模型输出，计算微平均 Sensitivity、Specificity 和 AUC。

    参数:
        labels (np.ndarray): 真实标签，形状 [N, C]，元素为 0/1。
        outputs_cls (np.ndarray): 模型输出1（如分类头），形状 [N, C]，概率值。
        outputs_prot (np.ndarray): 模型输出2（如原型头），形状 [N, C]，概率值。
        threshold (float): 概率转二进制标签的阈值（仅用于 Sensitivity/Specificity）。

    返回:
        dict: 包含两个输出的评估指标（Hamming Loss, Sensitivity, Specificity, AUC）。
    """
    # 计算单个输出的评估指标
    def compute_metrics(y_true, y_prob, y_pred=None):
        # 展平为微平均计算
        y_true_flat = y_true.flatten()
        y_prob_flat = y_prob.flatten()

        # Hamming Loss（需要二值化预测）
        if y_pred is None:
            y_pred = (y_prob > threshold).astype(int)
        y_pred_flat = y_pred.flatten()
        hl = hamming_loss(y_true, y_pred)

        # 微平均 Sensitivity (Recall) 和 Specificity
        TP = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
        TN = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
        FP = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
        FN = np.sum((y_true_flat == 1) & (y_pred_flat == 0))

        sensitivity = TP / (TP + FN + 1e-10)
        specificity = TN / (TN + FP + 1e-10)

        # AUC（无需二值化，直接基于概率）
        try:
            auc = roc_auc_score(y_true, y_prob, average="micro")
        except ValueError:
            auc = np.nan  # 处理无法计算的情况（如所有标签相同）

        return {
            "Total num": len(y_true),
            "hamming_loss": round(hl, 3),
            "sensitivity": round(sensitivity, 3),
            "specificity": round(specificity, 3),
            "auc": round(auc, 3),
            "f1": round(f1_score(y_true, y_pred, average='micro'), 3),
        }

    # 分别评估两个输出
    preds_cls = (outputs_cls > threshold).astype(int)
    metrics_cls = compute_metrics(labels, outputs_cls, preds_cls)

    preds_prot = (outputs_prot > threshold).astype(int)
    metrics_prot = compute_metrics(labels, outputs_prot, preds_prot)

    # 返回结果
    return {
        "micro_cls": metrics_cls,
        "micro_prot": metrics_prot,
    }

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1, )).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
    return total / num_samples

def sigmoid_rampup(current, rampup_length, exp_coe=5.0):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-exp_coe * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix = np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    print("Finish Generating Candidate Label Sets!\n")
    return partialY

def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

def generate_hierarchical_cv_candidate_labels(dataname, train_labels, partial_rate=0.1):
    assert dataname == 'cifar100'

    meta = unpickle('data/cifar-100-python/meta')

    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
    label2idx = {fine_label_names[i]:i for i in range(100)}

    x = '''aquatic mammals#beaver, dolphin, otter, seal, whale
fish#aquarium fish, flatfish, ray, shark, trout
flowers#orchid, poppy, rose, sunflower, tulip
food containers#bottle, bowl, can, cup, plate
fruit and vegetables#apple, mushroom, orange, pear, sweet pepper
household electrical devices#clock, keyboard, lamp, telephone, television
household furniture#bed, chair, couch, table, wardrobe
insects#bee, beetle, butterfly, caterpillar, cockroach
large carnivores#bear, leopard, lion, tiger, wolf
large man-made outdoor things#bridge, castle, house, road, skyscraper
large natural outdoor scenes#cloud, forest, mountain, plain, sea
large omnivores and herbivores#camel, cattle, chimpanzee, elephant, kangaroo
medium-sized mammals#fox, porcupine, possum, raccoon, skunk
non-insect invertebrates#crab, lobster, snail, spider, worm
people#baby, boy, girl, man, woman
reptiles#crocodile, dinosaur, lizard, snake, turtle
small mammals#hamster, mouse, rabbit, shrew, squirrel
trees#maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
vehicles 1#bicycle, bus, motorcycle, pickup truck, train
vehicles 2#lawn_mower, rocket, streetcar, tank, tractor'''

    x_split = x.split('\n')
    hierarchical = {}
    reverse_hierarchical = {}
    hierarchical_idx = [None] * 20
    # superclass to find other sub classes
    reverse_hierarchical_idx = [None] * 100
    # class to superclass
    super_classes = []
    labels_by_h = []
    for i in range(len(x_split)):
        s_split = x_split[i].split('#')
        super_classes.append(s_split[0])
        hierarchical[s_split[0]] = s_split[1].split(', ')
        for lb in s_split[1].split(', '):
            reverse_hierarchical[lb.replace(' ', '_')] = s_split[0]
            
        labels_by_h += s_split[1].split(', ')
        hierarchical_idx[i] = [label2idx[lb.replace(' ', '_')] for lb in s_split[1].split(', ')]
        for idx in hierarchical_idx[i]:
            reverse_hierarchical_idx[idx] = i

    # end generate hierarchical
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate
    mask = np.zeros_like(transition_matrix)
    for i in range(len(transition_matrix)):
        superclass = reverse_hierarchical_idx[i]
        subclasses = hierarchical_idx[superclass]
        mask[i, subclasses] = 1

    transition_matrix *= mask
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    print("Finish Generating Candidate Label Sets!\n")
    return partialY
