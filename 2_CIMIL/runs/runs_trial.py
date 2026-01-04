import os
import re
from numpy import test
import torch
import tqdm
import sys
import pandas as pd
import csv

import cv2
import torch.nn as nn
import torch.cuda as cuda
import torch.distributed as dist
import torch.nn.functional as F

from torch.optim import Adam, SGD
from datetime import datetime
import json

from utils.utils import *

def train(model, center_loss_model, center_loss_img_model, scheduler, optimizer, criterion, train_dataset, train_loader, train_sampler, test_loader, optimizer_center, optimizer_center_img, 
  scheduler_center, scheduler_center_img, patient_label_map, checkpoint_dir, opt, args):

    model_save_dir = os.path.join(checkpoint_dir,'model')
    os.makedirs(model_save_dir, exist_ok=True)

    for epoch in range(1, opt.epoch + 1):
      # 随机选择负样本，保证样本平衡
      train_dataset.random_choose_nocancer_img() 
      if args.local_rank <= 0:
        print(f'\n#-------- Epoch {epoch} training --------- #')
        print("Epoch:{:03d}/{:03d}".format(epoch, opt.epoch), ", lr: ",
                optimizer.param_groups[0]['lr'],
                optimizer.param_groups[1]['lr'],
                optimizer.param_groups[2]['lr'])
      if args.device_num > 1:
        train_sampler.set_epoch(epoch)

      model.train()
      for i, (bags, labels, names) in enumerate(test_loader, start=1):
        ## feature extraction
        _, max_instances, _, _, _ = bags.size() # bags: b, k, c, w, h
        bag_img_feature = []
        if max_instances < opt.topk:
          bags = torch.cat((bags, torch.zeros(bags.size(0), opt.topk - max_instances, bags.size(2), bags.size(3), bags.size(4))),
                              dim=1)
          labels = torch.cat((labels, torch.zeros(labels.size(0), opt.topk - max_instances)), dim=1)
          names.append(["zeros"]*(opt.topk - max_instances))
        for instance in range(0, max_instances, opt.instance_batch_size):
          split_images = bags[:, instance:instance+opt.instance_batch_size, :, :, :]
          # split_images = split_images.squeeze(0)
          split_images = split_images.cuda()

          img_features = model.feature_extractor.forward_features(split_images)
          bag_img_feature.append(img_features.squeeze(0))
        bag_img_feature = torch.cat(bag_img_feature, dim=0) # b, k, E
        
        ## feature aggregation
        