import torch
import numpy as np
from thop import profile
from thop import clever_format
from scipy.ndimage import map_coordinates

from torch.optim.lr_scheduler import _LRScheduler


class PolyLr(_LRScheduler):
    def __init__(self, optimizer, gamma, max_iteration, minimum_lr=0, warmup_iteration=0, last_epoch=-1):
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.minimum_lr = minimum_lr
        self.warmup_iteration = warmup_iteration

        super(PolyLr, self).__init__(optimizer, last_epoch)

    def poly_lr(self, base_lr, step):
        return (base_lr - self.minimum_lr) * ((1 - (step / self.max_iteration)) ** self.gamma) + self.minimum_lr

    def warmup_lr(self, base_lr, alpha):
        return base_lr * (1 / 10.0 * (1 - alpha) + alpha)

    # def get_lr(self):
    #     if self.last_epoch < self.warmup_iteration:
    #         alpha = self.last_epoch / self.warmup_iteration
    #         lrs = [min(self.warmup_lr(base_lr, alpha), self.poly_lr(base_lr, self.last_epoch)) for base_lr in
    #                 self.base_lrs]
    #     else:
    #         lrs = [self.poly_lr(base_lr, self.last_epoch) for base_lr in self.base_lrs]

    #     return lrs
    
    def get_lr(self):
        if self.last_epoch < 20:
            lrs = self.base_lrs
        elif self.last_epoch == 20:
            lrs = [base_lr * 0.1 for base_lr in self.base_lrs]
        else:
            lrs = [base_lr / (10 ** (self.last_epoch // 70 + 1)) for base_lr in self.base_lrs]
        return lrs
    
class PolyLr_LSTM(_LRScheduler):
    def __init__(self, optimizer, gamma, max_iteration, minimum_lr=0, warmup_iteration=0, last_epoch=-1):
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.minimum_lr = minimum_lr
        self.warmup_iteration = warmup_iteration

        super(PolyLr_LSTM, self).__init__(optimizer, last_epoch)

    # def get_lr(self):
    #     lrs = self.base_lrs
    #     return lrs

    def get_lr(self):
        # decay 0.5 each 50 epochs
        decay = 0.5 ** (self.last_epoch // 50)
        lrs = [base_lr * decay for base_lr in self.base_lrs]
        return lrs


    # def get_lr(self):
    #     if self.last_epoch < 18:
    #         lrs = self.base_lrs
    #     elif self.last_epoch >= 18 and self.last_epoch < 28:
    #         lrs = [base_lr / 3 for base_lr in self.base_lrs]
    #     elif self.last_epoch >= 28 and self.last_epoch < 50:
    #         lrs = [base_lr / 10 for base_lr in self.base_lrs]
    #     else:
    #         lrs = [base_lr / 30 for base_lr in self.base_lrs]
    #     return lrs
    
    # def get_lr(self):
    #     if self.last_epoch < 7:
    #         lrs = self.base_lrs
    #     elif self.last_epoch >= 7 and self.last_epoch < 10:
    #         lrs = [base_lr / 3 for base_lr in self.base_lrs]
    #     elif self.last_epoch >= 10 and self.last_epoch < 15:
    #         lrs = [base_lr / 10 for base_lr in self.base_lrs]
    #     else:
    #         lrs = [base_lr / 30 for base_lr in self.base_lrs]
    #     return lrs

    # def get_lr(self):
    #     if self.last_epoch < 15:
    #         lrs = self.base_lrs
    #     # elif self.last_epoch >= 15 and self.last_epoch < 30:
    #     #     lrs = [base_lr * 0.1 for base_lr in self.base_lrs]
    #     elif self.last_epoch >= 15 and self.last_epoch < 25:
    #         lrs = [base_lr / 3 for base_lr in self.base_lrs]
    #     elif self.last_epoch >= 25 and self.last_epoch < 35:
    #         lrs = [base_lr / 10 for base_lr in self.base_lrs]
    #     else:
    #         lrs = [base_lr / 100 for base_lr in self.base_lrs]
    #         # lrs = [base_lr / (10 ** (self.last_epoch // 60 + 2)) for base_lr in self.base_lrs]
    #     return lrs
    
    # def get_lr(self):
    #     if self.last_epoch < 15:
    #         lrs = self.base_lrs
    #     # elif self.last_epoch >= 15 and self.last_epoch < 30:
    #     #     lrs = [base_lr * 0.1 for base_lr in self.base_lrs]
    #     elif self.last_epoch >= 15 and self.last_epoch < 20:
    #         lrs = [base_lr / 3 for base_lr in self.base_lrs]
    #     elif self.last_epoch >= 20 and self.last_epoch < 35:
    #         lrs = [base_lr / 10 for base_lr in self.base_lrs]
    #     else:
    #         lrs = [base_lr / 100 for base_lr in self.base_lrs]
    #         # lrs = [base_lr / (10 ** (self.last_epoch // 60 + 2)) for base_lr in self.base_lrs]
    #     return lrs
    
    # def get_lr(self):
    #     if self.last_epoch < 5:
    #         lrs = self.base_lrs
    #     else:
    #         lrs = [base_lr / (1+self.last_epoch/15) for base_lr in self.base_lrs]
    #     return lrs
    

class PolyLr_ADMIL(_LRScheduler):
    def __init__(self, optimizer, gamma, max_iteration, minimum_lr=0, warmup_iteration=0, last_epoch=-1):
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.minimum_lr = minimum_lr
        self.warmup_iteration = warmup_iteration

        super(PolyLr_ADMIL, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < 20:
            lrs = self.base_lrs
        elif self.last_epoch >= 20 and self.last_epoch < 30:
            lrs = [base_lr * 0.3 for base_lr in self.base_lrs]
        else:
            lrs = [base_lr / 10 for base_lr in self.base_lrs]
            # lrs = [base_lr / (10 ** (self.last_epoch // 60 + 1)) for base_lr in self.base_lrs]
        return lrs


class PolyLr_LSTM_finetune(_LRScheduler):
    def __init__(self, optimizer, gamma, max_iteration, minimum_lr=0, warmup_iteration=0, last_epoch=-1):
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.minimum_lr = minimum_lr
        self.warmup_iteration = warmup_iteration

        super(PolyLr_LSTM_finetune, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < 5:
            lrs = [base_lr / 10 for base_lr in self.base_lrs]
        elif self.last_epoch >= 5:
            lrs = [base_lr / 100 for base_lr in self.base_lrs]
        return lrs