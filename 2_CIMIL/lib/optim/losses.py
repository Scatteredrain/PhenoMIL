import torch
import torch.nn.functional as F

def bce_iou_loss(pred, mask):
    # 如果一个mask全黑，需要增大分割权重，尽量让他分割不出来
    # 全黑判断可能需要判断maks有多少个像素点，需要遍历一下所有mask，看看每个mask的像素点有多少
    foreground_pixels = mask.sum(dim=(2, 3)).float()
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    
    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)

    weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    weighted_iou = (weight * iou).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    loss = weighted_bce + weighted_iou

    # threshold = 1000.0
    # scale = torch.clamp(threshold / (foreground_pixels + 1e-6), max=1.0)
    mask_cancer = foreground_pixels > 100
    loss_cancer = loss[mask_cancer].mean() if mask_cancer.any() else torch.tensor(0.0).cuda()
    mask_nocancer = foreground_pixels <= 100
    loss_nocancer = loss[mask_nocancer].mean() if mask_nocancer.any() else torch.tensor(0.0).cuda()
    # loss_mean = loss[mask_nocancer].mean() * loss[mask_nocancer].size(0) / loss.size(0) \
    #     + loss[mask_cancer].mean() * loss[mask_cancer].size(0) / loss.size(0)
    # print(loss.mean(), loss_mean)

    return loss.mean(), loss_cancer, loss_nocancer, \
        loss[mask_cancer].size(0), loss[mask_nocancer].size(0), loss[mask_cancer].size(0) + loss[mask_nocancer].size(0)

def dice_bce_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    
    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (2. * inter + 1) / (union + 1)

    return (bce + iou).mean()

def tversky_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    pred = torch.sigmoid(pred)       

    #flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    #True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()    
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)  

    return (1 - Tversky) ** gamma

def tversky_bce_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

    pred = torch.sigmoid(pred)       

    #flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    #True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()    
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)  

    return bce + (1 - Tversky) ** gamma