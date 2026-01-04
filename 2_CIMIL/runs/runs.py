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

def train(model, center_loss_model, center_loss_img_model, scheduler, optimizer, criterion, train_dataset, train_loader, train_sampler, test_loader, optimizer_center, optimizer_center_img, scheduler_center, scheduler_center_img, patient_label_map, checkpoint_dir, opt, args):
    if args.local_rank <= 0:
        auc_best = 0.0
        best_epoch = 0
        csv_result = []
        csv_result_test = []
        columns = ['epoch', 'loss', 'loss img', 'loss pat', 'loss center pat', 'loss center img',
                    'total_img', 'corr_img', 'TP_img', 'FP_img', 'TN_img', 'FN_img',
                    'acc_img', 'rec_img', 'pre_img', 'spe_img', 'f1_img', 'auc_img',
                    'total_img_center', 'corr_img_center', 'TP_img_center', 'FP_img_center', 'TN_img_center', 'FN_img_center',
                    'acc_img_center', 'rec_img_center', 'pre_img_center', 'spe_img_center', 'f1_img_center', 'auc_img_center',
                    'total_pat', 'corr_pat', 'TP_pat', 'FP_pat', 'TN_pat', 'FN_pat',
                    'acc_pat', 'rec_pat', 'pre_pat', 'spe_pat', 'f1_pat', 'auc_pat',
                    'total_img_mean', 'corr_img_mean', 'TP_img_mean', 'FP_img_mean', 'TN_img_mean', 'FN_img_mean',
                    'acc_img_mean', 're_img_mean', 'pre_img_mean', 'spe_img_mean', 'f1_img_mean', 'auc_img_mean',
                    'total_pat_center', 'corr_pat_center', 'TP_pat_center', 'FP_pat_center', 'TN_pat_center', 'FN_pat_center',
                    'acc_pat_center', 're_pat_center', 'pre_pat_center', 'spe_pat_center', 'f1_pat_center', 'auc_pat_center',
                    'total_img_center_mean', 'corr_img_center_mean', 'TP_img_center_mean', 'FP_img_center_mean', 'TN_img_center_mean', 'FN_img_center_mean',
                    'acc_img_center_mean', 're_img_center_mean', 'pre_img_center_mean', 'spe_img_center_mean', 'f1_img_center_mean', 'auc_img_center_mean']
        csv_result.append(columns)
        csv_result_test.append(columns)

    model_save_dir = os.path.join(checkpoint_dir,'model')
    os.makedirs(model_save_dir, exist_ok=True)

    epoch_iter = range(1, opt.epoch + 1)
    for epoch in epoch_iter:
        # 随机选择负样本，保证样本平衡
        train_dataset.random_choose_nocancer_img() 
        if args.local_rank <= 0:
            print(f'\n#-------- Epoch {epoch} training --------- #')
            print("Epoch:{:03d}/{:03d}".format(epoch, len(epoch_iter)), ", lr: ",
                    optimizer.param_groups[0]['lr'],
                    optimizer.param_groups[1]['lr'],
                    optimizer.param_groups[2]['lr'])
        if args.device_num > 1:
            train_sampler.set_epoch(epoch)

        # if args.local_rank <= 0 and args.verbose is True:
        #     step_iter = tqdm.tqdm(enumerate(train_loader, start=1), desc='Iter', total=len(
        #         train_loader), position=1, leave=True, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}')
        # else:
        #     step_iter = enumerate(train_loader, start=1)
        # step_iter = enumerate(train_loader, start=1)
        # ---- eval train dataset ----
        train_inference_result, batch_topk_imgs, batch_topk_imgs_label, batch_topk_names = val(opt, args, train_loader, model, epoch, patient_label_map, center_loss_model,
                center_loss_img_model,checkpoint_dir,'train')

        # ---- train ----
        model.train()
        if opt.is_center_loss:
            center_loss_model.train()
        if opt.is_img_center_loss:
            center_loss_img_model.train()

        epoch_loss = {"total_loss": 0., "img_loss": 0., "pat_loss": 0.,
                        "pat_center_loss": 0., "img_center_loss": 0.}
        train_dict = {"correct": 0, "total": 0, "TP": 0, "FP": 0, "TN": 0, "FN": 0, "all_labels": [], "all_probs": []}

        batch_topk_imgs = torch.stack(batch_topk_imgs, dim=0) # patient_num*k*c*h*w
        batch_topk_imgs_label = torch.stack(batch_topk_imgs_label, dim=0) # patient_num*k*c*h*w

        indices = torch.randperm(batch_topk_imgs.size(0))
        batch_topk_imgs = batch_topk_imgs[indices]
        batch_topk_imgs_label = batch_topk_imgs_label[indices]
        batch_topk_names = [batch_topk_names[idx] for idx in indices]

        batch_pat_label = [label[0] for label in batch_topk_imgs_label]
        batch_pat_label = torch.stack(batch_pat_label, dim=0)

        total_pat, _, _, _, _ = batch_topk_imgs.size()

        if args.local_rank <= 0 and args.verbose is True:
            # train_iter = tqdm.tqdm(range(0, total_img, 16), desc='Iter', total=len(
            #     range(0, total_img, 16)), position=1, leave=True, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:20}{r_bar}')
            train_iter = range(0, total_pat, opt.train_batch_size)
        else:
            train_iter = range(0, total_pat, opt.train_batch_size)

        total_step = 0
        for instance in train_iter:
            split_pats = batch_topk_imgs[instance:instance+opt.train_batch_size, :, :, :, :]
            if split_pats.size(0) < opt.train_batch_size:
                continue

            if opt.lstm_random:
                # 对第二个维度进行独立打乱
                indices = torch.stack([torch.randperm(split_pats.size(1)) for _ in range(split_pats.size(0))])
                split_pats = torch.stack([t[i] for t, i in zip(split_pats, indices)])
            split_pats = split_pats.cuda()
            split_labels = batch_topk_imgs_label[instance:instance+opt.train_batch_size].cuda()
            split_names = batch_topk_names[instance:instance+opt.train_batch_size]
            split_pat_labels = batch_pat_label[instance:instance+opt.train_batch_size].cuda()

            optimizer.zero_grad()
            if opt.is_center_loss:
                optimizer_center.zero_grad()
            if opt.is_img_center_loss:
                optimizer_center_img.zero_grad()
            img_feature, x_fc, bag_feature, lstm_out = model(split_pats, is_train=True)
            loss_img = criterion(x_fc.view(-1, x_fc.size(-1)), split_labels.view(-1))
            loss_pat = criterion(lstm_out, split_pat_labels)
            loss = loss_img + loss_pat * 0.5
            if opt.is_center_loss:
                loss_center = center_loss_model(bag_feature, split_pat_labels)
                loss += loss_center
            else:
                loss_center = torch.tensor(0.)
            if opt.is_img_center_loss:
                loss_center_img = center_loss_img_model(img_feature.view(-1, img_feature.size(-1)), split_labels.view(-1))
                loss += loss_center_img
            else:
                loss_center_img = torch.tensor(0.)
            loss.backward()
            optimizer.step()
            if opt.is_center_loss:
                optimizer_center.step()
            if opt.is_img_center_loss:
                optimizer_center_img.step()
            total_step += 1

            _, predicted = torch.max(lstm_out.data, 1)
            sample = {"label": split_pat_labels}
            epoch_loss["total_loss"] += loss.item()
            epoch_loss["img_loss"] += loss_img.item()
            epoch_loss["pat_loss"] += loss_pat.item()
            epoch_loss["pat_center_loss"] += loss_center.item()
            epoch_loss["img_center_loss"] += loss_center_img.item()

            train_dict = result_process(
                    train_dict, predicted, sample['label'],
                    split_pats.size(0), split_pat_labels.data.cpu().numpy(), F.softmax(lstm_out, 1).data[:,1].cpu().numpy())
            del split_pats
            # # 强制进行垃圾回收
            # gc.collect()
            # # 清空未使用的缓存内存
            # torch.cuda.empty_cache()

        dist.barrier()
        epoch_loss["total_loss"] /= total_step
        epoch_loss["img_loss"] /= total_step
        epoch_loss["pat_loss"] /= total_step
        epoch_loss["pat_center_loss"] /= total_step
        epoch_loss["img_center_loss"] /= total_step
        if args.device_num > 1:
            epoch_loss_merged = merge_dicts_across_processes(epoch_loss)
            train_dict_merged = merge_dicts_across_processes(train_dict)

        if args.local_rank <= 0:

            print(
                "[{}] \nEpoch:{:03d}/{:03d},Loss:{:.4f},Loss Img:{:.4f},Loss Pat:{:.4f},Loss CenPat:{:.4f},Loss CenImg:{:.4f}"
                .format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch,
                    len(epoch_iter),
                    epoch_loss_merged["total_loss"] / args.device_num,
                    epoch_loss_merged["img_loss"] / args.device_num,
                    epoch_loss_merged["pat_loss"] / args.device_num,
                    epoch_loss_merged["pat_center_loss"] / args.device_num,
                    epoch_loss_merged["img_center_loss"] / args.device_num,
                ))
            train_inference_result = [epoch, epoch_loss_merged["total_loss"] / args.device_num, epoch_loss_merged["img_loss"] / args.device_num,
                            epoch_loss_merged["pat_loss"] / args.device_num, epoch_loss_merged["pat_center_loss"] / args.device_num,
                            epoch_loss_merged["img_center_loss"] / args.device_num] + train_inference_result
            csv_result.append(train_inference_result)
            print_result("Train Stage:", train_dict_merged)
            
        # ---- test ----
        print(f"\n#-------- Epoch {epoch} Val Result on testdataset ---------#")
        test_result = val(opt, args, test_loader, model, epoch, patient_label_map, center_loss_model,
                        center_loss_img_model,checkpoint_dir,'test')
        if args.local_rank <= 0:
            csv_result_test.append(test_result)
            auc_now = np.max([test_result[41], test_result[53], test_result[65], test_result[77]])
            if auc_now > auc_best:
                auc_best = auc_now
                best_epoch = epoch
            
                torch.save(
                    model.module.state_dict() if args.device_num > 1 else model.state_dict(),
                    os.path.join(model_save_dir, "Best_epoch_" + str(epoch) + ".pth"),
                )
                if opt.is_center_loss:
                    torch.save(
                        center_loss_model.module.state_dict() if args.device_num > 1 else center_loss_model.state_dict(),
                        os.path.join(model_save_dir, "Best_epoch_" + str(epoch) + "_centerloss.pth"),
                    )
                if opt.is_img_center_loss:
                    torch.save(
                        (
                            center_loss_img_model.module.state_dict()
                            if args.device_num > 1
                            else center_loss_img_model.state_dict()
                        ),
                        os.path.join(model_save_dir, "Best_epoch_" + str(epoch) + "_centerlossimg.pth"),
                    )
            else:
                torch.save(
                    model.module.state_dict() if args.device_num > 1 else model.state_dict(),
                    os.path.join(model_save_dir, "epoch_" + str(epoch) + ".pth"),
                )
                if opt.is_center_loss:
                    torch.save(
                        center_loss_model.module.state_dict() if args.device_num > 1 else center_loss_model.state_dict(),
                        os.path.join(model_save_dir, "epoch_" + str(epoch) + "_centerloss.pth"),
                    )
                if opt.is_img_center_loss:
                    torch.save(
                        (
                            center_loss_img_model.module.state_dict()
                            if args.device_num > 1
                            else center_loss_img_model.state_dict()
                        ),
                        os.path.join(model_save_dir, "epoch_" + str(epoch) + "_centerlossimg.pth"),
                    )
            print(f"\n#-------- Best result: Epoch {best_epoch} | Auc {auc_best} ---------#")

        scheduler.step()
        if opt.is_center_loss:
            scheduler_center.step()
        if opt.is_img_center_loss:
            scheduler_center_img.step()
        # 储存结果
        if args.local_rank <= 0:
            filename_train =  os.path.join(checkpoint_dir, "train_result.csv")
            filename_test =  os.path.join(checkpoint_dir, "test_result.csv")
            # 使用with语句打开文件，这样可以确保文件在操作完成后被正确关闭
            with open(filename_train, 'w', newline='', encoding='utf-8') as csvfile:
                # 创建csv.writer对象
                writer = csv.writer(csvfile)
                # 遍历列表，将每行数据写入CSV文件
                for row in csv_result:
                    writer.writerow(row)
            with open(filename_test, 'w', newline='', encoding='utf-8') as csvfile:
                # 创建csv.writer对象
                writer = csv.writer(csvfile)
                # 遍历列表，将每行数据写入CSV文件
                for row in csv_result_test:
                    writer.writerow(row)

def val(opt, args, test_loader, model, epoch, patient_label_map,
        model_center_loss, model_center_loss_img, checkpoint_dir, mode):

    epoch_result = []
    batch_topk_imgs = []
    batch_topk_imgs_label = []
    batch_topk_names = []
    model.eval()
    if opt.is_center_loss:
        model_center_loss.eval()
    if opt.is_img_center_loss:
        model_center_loss_img.eval()

    epoch_loss = {"total_loss": 0.0, "img_loss": 0.0, "pat_loss": 0.0, "pat_center_loss": 0.0, "img_center_loss": 0.0}
    img_dict = {"correct": 0, "total": 0, "TP": 0, "FP": 0, "TN": 0, "FN": 0, "all_labels": [], "all_probs": []}
    img_center_dict = {"correct": 0, "total": 0, "TP": 0, "FP": 0, "TN": 0, "FN": 0, "all_labels": [], "all_probs": []}
    pat_lstm_dict = {"correct": 0, "total": 0, "TP": 0, "FP": 0, "TN": 0, "FN": 0, "all_labels": [], "all_probs": []}
    pat_img_mean_dict = {"correct": 0, "total": 0, "TP": 0, "FP": 0, "TN": 0, "FN": 0, "all_labels": [], "all_probs": []}
    pat_center_dict = {"correct": 0, "total": 0, "TP": 0, "FP": 0, "TN": 0, "FN": 0, "all_labels": [], "all_probs": []}
    pat_img_center_mean_dict = {"correct": 0, "total": 0, "TP": 0, "FP": 0, "TN": 0, "FN": 0, "all_labels": [], "all_probs": []}

    total_step = 0
    step_iter = enumerate(test_loader, start=1)
    bag_save_dir = os.path.join(checkpoint_dir,
                           'debug', "test", "bags")
    os.makedirs(bag_save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (bags, labels, names) in step_iter:
            _, max_instances, _, _, _ = bags.size()
            bag_softmax_out = []
            bag_img_feature = []
            if max_instances < opt.topk:
                for _ in range(0, opt.topk - max_instances):
                    bags = torch.cat((bags, torch.zeros(topk_imgs.size(1), topk_imgs.size(2), topk_imgs.size(3)).unsqueeze(0).unsqueeze(0)),
                                        dim=1)
                    labels = torch.cat((labels, torch.tensor(0).unsqueeze(0).unsqueeze(0)), dim=1)
                    names.append(["zeros"])

            for instance in range(0, max_instances, opt.instance_batch_size):
                split_images = bags[:, instance:instance+opt.instance_batch_size, :, :, :]
                # split_images = split_images.squeeze(0)
                split_images = split_images.cuda()

                img_features, _, split_softmax_out = model(split_images, is_train=False)
                bag_softmax_out.append(split_softmax_out.squeeze(0))
                bag_img_feature.append(img_features.squeeze(0))
                split_images = split_images.cpu()

            bag_softmax_out = torch.cat(bag_softmax_out, dim=0)
            probs_class_1 = bag_softmax_out[:, 1]

            topk_values, topk_indices = torch.topk(probs_class_1, min(probs_class_1.size(0), opt.topk))
            topk_indices = topk_indices.to(bags.device)

            topk_imgs = bags[:, topk_indices, :, :, :]
            topk_labels = labels[:, topk_indices]
            topk_names = [names[idx] for idx in topk_indices.data]

            img_feature, x_fc, bag_feature, lstm_out = model(topk_imgs.cuda(), is_train=True)

            if mode == 'test':
                criterion = nn.CrossEntropyLoss()
                loss_img = criterion(x_fc.view(-1, x_fc.size(-1)).cpu(), topk_labels.view(-1))
                loss_pat = criterion(lstm_out.cpu(), topk_labels[0][0].clone().detach().unsqueeze(0))
                loss = loss_img + loss_pat * 0.5
                if opt.is_center_loss:
                    loss_center = model_center_loss(bag_feature, topk_labels[0][0].clone().detach().unsqueeze(0).cuda())
                    loss += loss_center.cpu()
                else:
                    loss_center = torch.tensor(0.)
                if opt.is_img_center_loss:
                    loss_center_img = model_center_loss_img(img_feature.view(-1, img_feature.size(-1)), topk_labels.view(-1).cuda())
                    loss += loss_center_img.cpu()
                else:
                    loss_center_img = torch.tensor(0.)
                epoch_loss["total_loss"] += loss.item()
                epoch_loss["img_loss"] += loss_img.item()
                epoch_loss["pat_loss"] += loss_pat.item()
                epoch_loss["pat_center_loss"] += loss_center.item()
                epoch_loss["img_center_loss"] += loss_center_img.item()
                total_step += 1

            topk_imgs = topk_imgs.squeeze(0)
            topk_labels = topk_labels.squeeze(0)

            if mode == 'train':
                batch_topk_imgs.append(topk_imgs)
                batch_topk_imgs_label.append(topk_labels)
                batch_topk_names.append(topk_names)

            patient_pred_lstm = torch.tensor([1]) if F.softmax(lstm_out, dim=1)[-1][-1].item() > 0.5 else torch.tensor([0])
            patient_pred_img_mean = torch.tensor([1]) if topk_values.mean() > 0.5 else torch.tensor([0])
            topk_img_pred = (topk_values > 0.5).float().cpu()
            if opt.is_center_loss:
                center_loss_output = model_center_loss.module.get_assignment(bag_feature)
                center_loss_probs = center_loss_output.detach()[:, 1].clone()
            else:
                center_loss_probs = torch.tensor([0])
            patient_pred_pat_center = torch.tensor([1]) if center_loss_probs[0] > 0.5 else torch.tensor([0])

            if opt.is_img_center_loss:
                img_feature = torch.cat(bag_img_feature, dim=0)
                center_loss_img_output = model_center_loss_img.module.get_assignment(img_feature.squeeze(0))
                # center_loss_img_output.size():torch.Size([7,2])
                center_loss_img_probs = center_loss_img_output.detach()[:, 1].clone().cpu()
                # center_loss_img_output.size():torch.Size([7])
            else:
                center_loss_img_probs = torch.zeros_like(probs_class_1).cpu()
            topk_values_img_center, _ = torch.topk(center_loss_img_probs, min(center_loss_img_probs.size(0), opt.topk))
            topk_img_pred_img_center = (topk_values_img_center > 0.5).float()
            patient_pred_img_center_mean = torch.tensor([1]) if topk_values_img_center.mean() > 0.5 else torch.tensor([0])

            sample = {"label": topk_labels}
            img_dict = result_process(
                img_dict, topk_img_pred, sample['label'],
                topk_imgs.size(0), topk_labels.cpu().numpy(), topk_values.cpu().numpy())
            img_center_dict = result_process(
                img_center_dict, topk_img_pred_img_center, sample['label'],
                topk_imgs.size(0), topk_labels.cpu().numpy(), topk_values_img_center.cpu().numpy())
            pat_lstm_dict = result_process(
                pat_lstm_dict, patient_pred_lstm, topk_labels[0],
                1, [topk_labels[0].cpu().item()], [F.softmax(lstm_out, dim=1)[-1][-1].cpu().item()])
            pat_img_mean_dict = result_process(
                pat_img_mean_dict, patient_pred_img_mean, topk_labels[0],
                1, [topk_labels[0].cpu().item()], [topk_values.mean().cpu().item()])
            pat_center_dict = result_process(
                pat_center_dict, patient_pred_pat_center, topk_labels[0],
                1, [topk_labels[0].cpu().item()], [center_loss_probs[0].cpu().item()])
            pat_img_center_mean_dict = result_process(
                pat_img_center_mean_dict, patient_pred_img_center_mean, topk_labels[0],
                1, [topk_labels[0].cpu().item()], [topk_values_img_center.mean().cpu().item()])

            # save all imgs in one bag
            if args.debug and epoch % 10 == 1:
                flag = flag_process(
                        patient_pred_lstm, patient_pred_img_mean, patient_pred_pat_center,
                        patient_pred_img_center_mean, topk_labels[0].item(), opt)

                debout = debug_tile_lstm_topk_in_bag(
                    bags.squeeze(0), names, center_loss_img_probs.cpu().numpy(), probs_class_1.cpu().numpy(),
                    patient_label_map, topk_indices.cpu(), F.softmax(lstm_out, dim=1)[-1][-1].cpu().item(),
                    center_loss_probs[0].cpu().item(), center_pred_img=topk_values_img_center.mean().item())
                if "-" in names[0][0].split("_")[0].zfill(5):
                    save_img_name = os.path.join(
                        bag_save_dir,
                        "patient_" + names[0][0].split("-")[0].zfill(5) +
                        flag + "_epoch_" + str(epoch) + "_iter_" + str(i) +
                        "_rank_" + str(args.local_rank) + ".png",
                    )
                else:
                    save_img_name = os.path.join(
                        bag_save_dir,
                        "patient_" + names[0][0].split("_")[0].zfill(5) +
                        flag + "_epoch_" + str(epoch) + "_iter_" + str(i) +
                        "_rank_" + str(args.local_rank) + ".png",
                    )
                cv2.imwrite(save_img_name, debout)

        # sync
        if mode == 'test':
            epoch_loss["total_loss"] /= total_step
            epoch_loss["img_loss"] /= total_step
            epoch_loss["pat_loss"] /= total_step
            epoch_loss["pat_center_loss"] /= total_step
            epoch_loss["img_center_loss"] /= total_step
        dist.barrier()
        if args.device_num > 1:
            if mode == 'test':
                epoch_loss_merged = merge_dicts_across_processes(epoch_loss)

            img_dict_merged = merge_dicts_across_processes(img_dict)
            pat_lstm_dict_merged = merge_dicts_across_processes(pat_lstm_dict)
            pat_img_mean_dict_merged = merge_dicts_across_processes(pat_img_mean_dict)
            if opt.is_center_loss:
                pat_center_dict_merged = merge_dicts_across_processes(pat_center_dict)
            if opt.is_img_center_loss:
                img_center_dict_merged = merge_dicts_across_processes(img_center_dict)
                pat_img_center_mean_dict_merged = merge_dicts_across_processes(pat_img_center_mean_dict)

        if args.local_rank <= 0:
            if mode == 'test':
                print(
                    "Loss: {:.4f}, Loss Img: {:.4f}, Loss Pat: {:.4f}, Loss CenPat: {:.4f}, Loss CenImg: {:.4f}"
                    .format(
                        epoch_loss_merged["total_loss"] / args.device_num,
                        epoch_loss_merged["img_loss"] / args.device_num,
                        epoch_loss_merged["pat_loss"] / args.device_num,
                        epoch_loss_merged["pat_center_loss"] / args.device_num,
                        epoch_loss_merged["img_center_loss"] / args.device_num,
                    ))
                epoch_result = [epoch, epoch_loss_merged["total_loss"] / args.device_num, epoch_loss_merged["img_loss"] / args.device_num,
                                epoch_loss_merged["pat_loss"] / args.device_num, epoch_loss_merged["pat_center_loss"] / args.device_num,
                                epoch_loss_merged["img_center_loss"] / args.device_num] + epoch_result
            
            epoch_result.extend(print_result("TopK img:", img_dict_merged))
            if opt.is_img_center_loss:
                epoch_result.extend(print_result("TopK img center:", img_center_dict_merged))
            else:
                epoch_result.extend([0]*12)
            epoch_result.extend(print_result("Pat LSTM:", pat_lstm_dict_merged))
            epoch_result.extend(print_result("Pat img mean:", pat_img_mean_dict_merged))
            if opt.is_center_loss:
                epoch_result.extend(print_result("Pat Center:", pat_center_dict_merged))
            else:
                epoch_result.extend([0]*12)
            if opt.is_img_center_loss:
                epoch_result.extend(print_result("Pat img center mean:", pat_img_center_mean_dict_merged))
            else:
                epoch_result.extend([0]*12)
    if mode == 'train':
        return epoch_result, batch_topk_imgs, batch_topk_imgs_label, batch_topk_names
    return epoch_result    