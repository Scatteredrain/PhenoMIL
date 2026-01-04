import os
import torch
import tqdm
import sys
import pandas as pd
import csv
import random
import cv2
import torch.nn as nn
import torch.cuda as cuda
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime
import json
from torch.utils.tensorboard.writer import SummaryWriter
import logging
import time 

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

# from utils.dataloader_multicenter import PolypDataset
from utils.dataloader_multicenter_addImgLabel import PolypDataset
from lib.optim import PolyLr_LSTM
# from lib.pvtv2_lstm import *
from lib.HP_encoder_aggregator import MILModel, MILModel_combine
# from lib.res2net_v1b_base import *
# from lib.admil import *
from utils.utils import *
# from runs.runs import train
# from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
import gc
import numpy as np

label_info = read_json()
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

if __name__ == '__main__':
    args = parse_args()
    opt = load_config(args.config)
    if args.local_rank <= 0:
        print(json.dumps(opt, indent=4))

    writer_path = os.path.join(args.checkpoint_dir,'summary')
    os.makedirs(writer_path,exist_ok=True)

    logging.basicConfig(filename=os.path.join(args.checkpoint_dir,'train_log.log'),
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    global writer
    writer = SummaryWriter(writer_path)
    if args.local_rank <= 0:
        logging.info(json.dumps(opt, indent=4))

    if args.device_num > 1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.device_num)
    
    train_dataset = PolypDataset(
                    index_root = opt.Train.Dataset.index_root ,
                    transform_list=opt.transform_list,
                    is_transform=True, is_train=True, testsize=opt.transform_list.resize.size[0],debug=args.debug)
                    
    test_dataset = PolypDataset( #image_root = opt.Train.Dataset.image_root,
                        index_root = opt.Test.Dataset.index_root,
                        transform_list=opt.transform_list,
                        is_transform=False, is_train=False, testsize=opt.transform_list.resize.size[0])

    if args.device_num > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=opt.batch_size,
                                    shuffle=train_sampler is None,
                                    sampler=train_sampler,
                                    num_workers=opt.Dataloader.num_workers,
                                    pin_memory=opt.Dataloader.pin_memory,
                                    prefetch_factor=2, 
                                    drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=opt.val_batch_size,
                                    shuffle=False,
                                    sampler=test_sampler,
                                    num_workers=opt.Dataloader.val_num_workers,
                                    pin_memory=opt.Dataloader.pin_memory)
    opt.is_center_loss = (opt.is_center_loss and args.pooling != 'DSMIL')
    # model = MILModel(num_classes=2, k=5, split_size=8)
    # model_name = opt.Model.name
    opt.attention_constrain = opt.attention_constrain and args.finetune == 'PiCO' and args.pooling == 'HPMIL'
    print('use attention constrain:', opt.attention_constrain)
    model = MILModel_combine(opt, is_center_loss=opt.is_center_loss, is_img_center_loss=opt.is_img_center_loss, attention_constrain=opt.attention_constrain,\
        is_img_loss=opt.is_img_loss, instance_batch_size=opt.instance_batch_size,\
            aggregator=args.pooling, joint_learning=opt.joint_learning, \
                MaskDrop_threshold=opt.Model.MaskDrop_threshold)
   
    checkpoint_dir = os.path.join(args.checkpoint_dir)

   

    if args.finetune_path is not None:
        load_path = args.finetune_path
    else:
        load_path = opt.Model.PiCO_pretrained_path
    print('load_path', load_path)
    prefixes = ['encoder_q.encoder.']
    new_model_state_dict = model.state_dict()
    pretrained_model_state_dict = torch.load(load_path, map_location='cpu')
    pretrained_model_state_dict = pretrained_model_state_dict['state_dict']
    pretrained_model_state_dict = {k.replace('module.', ''): v for k, v in pretrained_model_state_dict.items()}
    pretrained_model_state_dict = {k if not any(k.startswith(prefix) for prefix in prefixes) else k.replace('encoder_q.encoder.', 'feature_extractor.'): v for k, v in pretrained_model_state_dict.items()}
    filtered_dict_polyp = {k: v for k, v in pretrained_model_state_dict.items() if k in new_model_state_dict and v.size() == new_model_state_dict[k].size()}
    new_model_state_dict.update(filtered_dict_polyp)
    model.load_state_dict(new_model_state_dict)

    if args.pooling == 'HPMIL':
        param_map = [
            # ('encoder_q.encoder.', 'feature_extractor.'),
            ('encoder_q.head.', 'aggregate.HP_mlp.'),
            ('encoder_q.fc.', 'aggregate.HP_head.')
        ]
        pretrained = pretrained_model_state_dict
        # for k, val in current.items():
        #     if 'HP_mlp' in k: print(k, val.shape)
        #     if 'HP_head' in k: print(k, val.shape)
        for k, val in pretrained_model_state_dict.items():
        #     if 'encoder_q.head' in k: print(k, val.shape)
        #     if 'encoder_q.fc' in k: print(k, val.shape)
            new_key = next((k.replace(src, dst) for src, dst in param_map if k.startswith(src)), k)
            if new_key in new_model_state_dict and val.shape == new_model_state_dict[new_key].shape:
                # if 'feature_extractor' not in new_key:  
                #     print(f"Mapping {k} to {new_key}")
                new_model_state_dict[new_key] = val

    model.load_state_dict(new_model_state_dict)


    print(f'## aggregator: {args.pooling}; finetune_method: {args.finetune}, frozen backbone: {args.frozen}; load_path: {load_path}')
    logging.info(f'## aggregator: {args.pooling}; finetune_method: {args.finetune}, frozen backbone: {args.frozen}; load_path: {load_path}')

    if args.local_rank <= 0:
        print('load keys:')
        print(filtered_dict_polyp.keys())
        print('\n########')

    if args.device_num > 1:
        find_unused_parameters = args.frozen or (args.finetune == 'PiCO' and args.pooling == 'HPMIL')
        print('Unused params finding:', find_unused_parameters)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=find_unused_parameters)
    else:
        model = model.cuda()

    centerloss_params = nn.ParameterList()
    backbone_params = nn.ParameterList()
    other_params = nn.ParameterList()

    for name, param in model.named_parameters():
        if 'center_loss' in name:
            centerloss_params.append(param)
        elif 'feature_extractor' in name:
            backbone_params.append(param)
            # if args.frozen:
            if args.frozen and args.finetune != None:
                param.requires_grad = False
        elif ('HP_mlp' in name) or ('HP_head' in name):
            param.requires_grad = False
            backbone_params.append(param)
        else:
            other_params.append(param)
    params_list = [{'params': other_params, 'lr': opt.lr}, 
                    {'params': backbone_params, 'lr': opt.lr*opt.backbone_lr}, 
                    {'params': centerloss_params, 'lr': opt.lr*opt.center_lr}]

    # params_list = model.parameters()
    optimizer = eval(opt.Optimizer.type)(
        params_list, opt.lr, weight_decay=opt.Optimizer.weight_decay)

    scheduler = eval(opt.Scheduler.type)(optimizer, gamma=opt.Scheduler.gamma,
                                            minimum_lr=opt.Scheduler.minimum_lr,
                                            max_iteration=len(train_loader) * opt.epoch,
                                            warmup_iteration=opt.Scheduler.warmup_iteration)

    criterion = nn.CrossEntropyLoss()
    # train(model, opt, args, fold)
    model_save_dir = os.path.join(checkpoint_dir,'model')
    os.makedirs(model_save_dir, exist_ok=True)
    best = 0.0
    best_epoch = 0

    # 启用混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    accumulation_steps = opt.accumulation_steps
    for epoch in range(1, opt.epoch + 1):

        if epoch == opt.Model.unfrozen_backbone_epoch and args.frozen and args.finetune != None:
            # unfrozen backbone
            print('##### unfrozen backbone')
            logging.info('##### unfrozen backbone')
            for name, param in model.named_parameters():
                if 'feature_extractor' in name:
                    param.requires_grad = True
                if 'HP_mlp' in name or 'HP_head' in name:
                    param.requires_grad = True
            
        # 随机选择负样本，保证样本平衡
        train_dataset.random_choose_nocancer_img() 
        if args.local_rank <= 0:
            print(f'\n#-------- Epoch {epoch} training --------- #')
            print("Epoch:{:03d}/{:03d}".format(epoch, opt.epoch), ", lr: ",
                    [optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))])
                    # optimizer.param_groups[1]['lr'])
                    # optimizer.param_groups[2]['lr'])
            # logging.info(f"Epoch:{epoch:03d}/{opt.epoch}, lr: {optimizer.param_groups[0]['lr']}")
            logging.info(f"Epoch:{epoch:03d}/{opt.epoch}, lr: {[optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))]}")

        if args.device_num > 1:
            train_sampler.set_epoch(epoch)

        ## train
        model.train()
        losses = 0.0
        # end = time.time()
        for i, (bags, bag_label, img_labels_8class, names) in enumerate(train_loader, start=1):
            
            _, max_instances, _, _, _ = bags.size() # bags: b=1, k, c, w, h
            
            with torch.cuda.amp.autocast():
                img_label = torch.repeat_interleave(bag_label, opt.topk, dim=0)

                out = model(bags, bag_label.cuda() if opt.is_center_loss else None, img_label.view(-1).cuda() if opt.is_img_loss else None, img_labels_8class.cuda() if opt.joint_learning else None, topk=opt.topk)
                
                bag_out = out['bag_out']
                # bag_out, importance_scores, aggregation_feature = model(bag_img_feature) # Nx1, NxK
                bag_out = F.softmax(bag_out, dim=1)
                loss_bag = criterion(bag_out, bag_label.cuda())
                # print(bag_out.shape, bag_label.shape)
                # select the topk important image features to calculate the image loss
                if opt.is_img_loss:

                    loss_img = out['img_loss']
                else:
                    loss_img = torch.tensor(0.0)
                
                if opt.is_center_loss:

                    loss_center_bag = out['center_loss']
                else:
                    loss_center_bag = torch.tensor(0.0)

                if opt.is_img_center_loss:
                    loss_center_img = out['center_loss_img']
                else:
                    loss_center_img = torch.tensor(0.0)

                if opt.attention_constrain:
                    loss_attention = out['loss_attention_constrain']
                else:
                    loss_attention = torch.tensor(0.0)

                if opt.joint_learning:
                    loss_joint = out['joint_learning_loss']
                else:
                    loss_joint = torch.tensor(0.0)

                loss = loss_bag + loss_img + loss_center_bag + loss_center_img + loss_attention + loss_joint

            loss = loss / accumulation_steps
            scaler.scale(loss).backward()  # 缩放梯度
            # check_gradient_layout(model) 
            del bag_out  # 及时释放中间变量
            # losses += loss.item()*accumulation_steps
            
            if (i+1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        
            losses += loss_bag.item()
    
            # print('Num_instances: ',max_instances)
            if args.local_rank <= 0 and i % (len(train_loader)//10) == 0:
                # print("Epoch:{:03d}/{:03d}".format(epoch, opt.epoch), ", Iter:{:03d}/{:03d}".format(i, len(train_loader)), ", loss: ", loss.item()*accumulation_steps)
                print("Epoch:{:03d}/{:03d}".format(epoch, opt.epoch), ", Iter:{:03d}/{:03d}".format(i, len(train_loader)), ", loss_bag: ", loss_bag.item(), ", loss_img: ", loss_img.item(),", loss_center_bag: ", loss_center_bag.item(), ", loss_center_img: ", loss_center_img.item(), ", loss_attention: ", loss_attention.item(), ", loss_joint: ", loss_joint.item())
                logging.info('Epoch:{:03d}/{:03d}, Iter:{:03d}/{:03d}, loss_bag: {:.4f}, loss_img: {:.4f}, loss_center_bag: {:.4f}, loss_center_img: {:.4f}, loss_attention: {:.4f}'.format(epoch, opt.epoch, i, len(train_loader), loss_bag.item(), loss_img.item(), loss_center_bag.item(), loss_center_img.item(), loss_attention.item(), loss_joint.item()))
                writer.add_scalars('loss', {'loss_bag': loss_bag.item(), 'loss_img': loss_img.item(), 'loss_center_bag': loss_center_bag.item(), 'loss_center_img': loss_center_img.item(), 'loss_attention': loss_attention.item(), 'loss_joint': loss_joint.item()}, epoch * len(train_loader) + i)

        losses = losses / len(train_loader)
        if args.local_rank <= 0:
            print("Epoch:{:03d}/{:03d}".format(epoch, opt.epoch), "Epoch Average loss: ", losses)
            logging.info("Epoch: {}, Average loss: {}".format(epoch, losses))
            writer.add_scalar('loss_bag_epoch', losses, epoch)

        ## eval
        if args.local_rank <= 0:
            print(f'\n#-------- Epoch {epoch} testing --------- #')
        model.eval()
        all_preds = []
        all_labels = []
        all_logits = []
        losses = []
        save_num = 0
        with torch.no_grad():
            for i, (bags, bag_label, img_labels_8class, names) in enumerate(test_loader, start=1):
                
                out = model(bags)
                bag_out = out['bag_out']
                bag_out = F.softmax(bag_out, dim=1)
                label_bag = bag_label.to(bag_out.device) # b
                bag_loss = criterion(bag_out, label_bag)

                all_labels.extend(label_bag.cpu().numpy())
                all_preds.extend(bag_out.argmax(dim=1).detach().cpu().numpy())
                all_logits.extend(bag_out[:, 1].detach().cpu().numpy())# 先收集到CPU
                losses.append(bag_loss.item())

                patient_name = names[0][0].split('_')[0]
                # if args.local_rank <= 0 and ((i % len(train_loader)//5) == 0 or patient_name in label_info):
                if args.local_rank <= 0 and save_num < 10 and patient_name in label_info and (epoch % 20 == 0) and random.random() < 0.5 \
                    and 'importance_scores' in out:
                    importance_scores = out['importance_scores']
                    if bag_out.argmax(dim=1).squeeze() == 1 and label_bag.squeeze() == 1:
                        flag = 'TP'
                    elif bag_out.argmax(dim=1).squeeze() == 1 and label_bag.squeeze() == 0:
                        flag = 'FP'
                    elif bag_out.argmax(dim=1).squeeze() == 0 and label_bag.squeeze() == 1:
                        flag = 'FN'
                    elif bag_out.argmax(dim=1).squeeze() == 0 and label_bag.squeeze() == 0:
                        flag = 'TN'

                    show_imgs = show_importance(bags, importance_scores, names, label_info)
                    save_dir = os.path.join(checkpoint_dir, 'imgs_eval')
                    os.makedirs(save_dir, exist_ok=True)
                    save_name = os.path.join(save_dir, f'{epoch}_{patient_name}_{flag}.jpg')
                    cv2.imwrite(save_name, show_imgs)
                    save_num+=1

        
        if args.device_num > 1:    # 分布式训练
            dist.barrier()
            cuda_logits = torch.tensor(all_logits, device=torch.cuda.current_device())
            cuda_preds = torch.tensor(all_preds, device=torch.cuda.current_device())
            cuda_labels = torch.tensor(all_labels, device=torch.cuda.current_device())
            cuda_losses = torch.tensor(losses, device=torch.cuda.current_device())
            all_logits = gather_lists(cuda_logits)
            all_preds = gather_lists(cuda_preds)
            all_labels = gather_lists(cuda_labels)
            losses = gather_lists(cuda_losses)
        # else:
        #     all_logits, all_preds, all_labels = [x.cpu() for x in all_logits], [x.cpu() for x in all_preds], [x.cpu() for x in all_labels]
        if args.local_rank <= 0:
            all_logits, all_preds, all_labels = np.array(all_logits), np.array(all_preds), np.array(all_labels)
            metrics = calculate_metrics(all_labels,all_preds,all_logits)
            metrics['CEloss'] = np.mean(losses)
            print(metrics)
            logging.info(metrics)
    
            auc = metrics['AUC']
            if auc > best:
                best_epoch = epoch
                best = auc
                # torch.save(model.state_dict(), save_path + 'PolypPVT.pth')
                torch.save(model.state_dict(), os.path.join(model_save_dir, '{}_auc_{}_best_mil.pth'.format(epoch,auc)))
                print('##############################################################################best, epoch', best, best_epoch)
                logging.info('##############################################################################best:{},epoch:{}'.format(best, best_epoch))
            else:
                # torch.save(model.state_dict(), os.path.join(model_save_dir, '{}_auc_{}_mil.pth'.format(epoch,auc)))
                print('best, epoch', best, best_epoch)
                logging.info('best:{},epoch:{}'.format(best, best_epoch))
            
            writer.add_scalars('metrics', {'AUC': metrics['AUC'], 'Acc': metrics['Acc']}, epoch)

        scheduler.step()

