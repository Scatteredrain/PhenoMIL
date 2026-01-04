
import argparse
import builtins
import math
import os
# import json
import random
import shutil
import time
import warnings
import torch
import torch.nn 
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import tensorboard_logger as tb_logger
import logging
import numpy as np
import torch.nn.functional as F
from lib.model import PiCO, concat_all_gather
# from model_multilabel_One_CL_Space_StableQueue import PiCO
# from model_multilabel_One_CL_Space import PiCO
from lib.resnet import *
from utils.utils_algo import *
from utils.utils_loss_multilabel import partial_loss, SupConLoss_One_CL_Space, partial_loss_DDP, \
    compute_jaccard_similarity_matrix, structure_loss, attention_constrain_loss, One_Bag_Loss
# from utils.endoscopy_HP_generatePseudoLabel_Seg import load_endoscopy
from utils.endoscopy_HP_dataset import load_polyp_seg_dataset, load_img_dataset, load_bag_dataset, generate_Class_Aware_thresholds, load_semi_img_dataset
from utils.utils_endoscopy import calculate_metrics, check_gradient_layout
# from scipy.spatial.distance import pdist, squareform

torch.set_printoptions(precision=2, sci_mode=False)

parser = argparse.ArgumentParser(description='PyTorch implementation of FPS-SL')
parser.add_argument('--dataset', default='endoscopy', type=str, 
                    choices=['endoscopy'],
                    help='dataset name')
parser.add_argument('--exp-dir', default='snapshots/debug', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('-a', '--arch', metavar='ARCH', default='pvt_v2_b2', choices=['resnet18','pvt_v2_b2'],
                    help='network architecture (encoder)')
parser.add_argument('-j', '--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=500, type=int, 
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr_decay_epochs', type=str, default='700,800,900',
                    help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10002', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=123, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--num-class', default=8, type=int,
                    help='number of class')
parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')
parser.add_argument('--moco_queue', default=8192, type=int, 
                    help='queue size; number of negative samples')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='momentum for updating momentum encoder')
parser.add_argument('--proto_m', default=0.99, type=float,
                    help='momentum for computing the momving average of prototypes')
parser.add_argument('--loss_weight', default=0.5, type=float,
                    help='contrastive loss weight')
parser.add_argument('--loss_weight_bag', default=0.5, type=float,
                    help='bag classification loss weight')
parser.add_argument('--conf_ema_range', default='0.95,0.8', type=str,
                    help='pseudo target updating coefficient (phi)')
parser.add_argument('--prot_start', default=0, type=int, 
                    help = 'Start Prototype Updating')
parser.add_argument('--semi_start', default=10000, type=int, 
                    help = 'Start semi-supervised learning, default for never')
parser.add_argument('--sup_bag_start', default=10000, type=int, 
                    help = 'Start bag-level supervised learning, including pseudo-bag sup and bag classification. default for never')
parser.add_argument('--dynamic_thre_epoch', default=20, type=int, 
                    help = 'updating dynamical threshold frequency')
parser.add_argument('--partial_rate', default=0.1, type=float, 
                    help='ambiguity level (q)')
parser.add_argument('--temperature', default=0.07, type=float, 
                    help='contrastive loss temperature')
parser.add_argument('--pretrained', default=True, type=bool,
                    help='use polyp-PVT pretrained model')
parser.add_argument('--pretrained_path', default='/mnt/data/yizhenyu/data/HP识别/workspace/PiCO/pretrained/PolypPVT.pth', type=str)
parser.add_argument('--train_constrain_no_neither_nums', type=int, default=500, 
                        help='constrain no neither nums, valid when train_no_neither=True')
parser.add_argument('--eval_constrain_no_neither_nums', type=int, default=0, 
                        help='constrain no neither nums, valid when train_no_neither=True')
parser.add_argument('--conf_thres', default=0.7, type=float,
                    help='unuse')
parser.add_argument('--pos_conf_percent', default=0.95, type=float, 
                    help='pos_per for unlabeled data')
parser.add_argument('--neg_conf_percent', default=0.95, type=float, 
                    help='neg_per for unlabeled data')
parser.add_argument('--mask_labeled_percent', default=0, type=float, 
                    help='mask_labeled_percent for unlabeled data')

############
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--hierarchical', action='store_true', 
                    help='for CIFAR-100 fine-grained training')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--add_bag_head', action='store_true')
parser.add_argument('--pseudo_bag_sup', action='store_true')
parser.add_argument('--add_seg_head', action='store_true')
parser.add_argument('--sample_balance', action='store_true')
parser.add_argument('--only_moco', action='store_true', default=False)
parser.add_argument('--no_queue', action='store_true', default=False)
parser.add_argument('--stable_queue', action='store_true', default=False)
parser.add_argument('--keep_moco', action='store_true', default=False)
parser.add_argument('--proto_cont', action='store_true', default=False)
parser.add_argument('--train_only_labeled', action='store_true', default=False, help='only train labeled data')
parser.add_argument('--train_no_both', action='store_true', default=False, help='the data labeled both pos and neg features will not be used')
# parser.add_argument('--semi_stage2', action='store_true', default=False, help='train semi stage2 with unlabeled data')
parser.add_argument('--only_bag', action='store_true', default=False, help='not train 8-label cls')
parser.add_argument('--label_enhancing', action='store_true', default=False, help='label_enhancing')
parser.add_argument('--switch_learning', action='store_true', default=False, help='switch_learning')
parser.add_argument('--train_no_neither', action='store_true', default=False, help='the data labeled neither pos or neg features will be used but ' \
                    'nums depend on train_constrain_no_neither_nums')
############
# parser.add_argument('--multiprocessing-distributed', default=True, type=bool)
# parser.add_argument('--hierarchical', default=False, type=bool)
# parser.add_argument('--debug', default=True, type=bool)
# parser.add_argument('--add_bag_head', default=False, type=bool)
# parser.add_argument('--only_moco', default=False, type=bool)
# parser.add_argument('--train_only_labeled', default=True, type=bool)
# parser.add_argument('--train_no_both', default=False, type=bool)
# parser.add_argument('--train_no_neither', default=True, type=bool)
# parser.add_argument('--cosine', default=True, type=bool,
#                     help='use cosine lr schedule')
############
parser.add_argument('--cls_task', default='multi_label', type=str, choices=['multi_cls', 'multi_label'],
                    help='classification task')
parser.add_argument('--train_file', default='./configs/multi_center_training/train_data.txt', type=str)
parser.add_argument('--train_bag_file', default='./configs/multi_center_training/train_data.txt', type=str)
parser.add_argument('--test_file', default='./configs/multi_center_training/eval_data.txt', type=str)
parser.add_argument('--CL_LossType', default='v4', type=str, choices=['v1', 'v2', 'v3','v4'],
                    help='CL loss type')
parser.add_argument('--Cls_LossType', default='v2', type=str, choices=['v1', 'v2'],
                    help='Cls loss type')   
parser.add_argument('--aggregate', default='v1', type=str, choices=['v1', 'v2'],
                    help='aggregator type')    
parser.add_argument('--proto_scoring', default='v2', type=str, choices=['v1', 'v2'],
                    help='protype scoring type')               
parser.add_argument('--topk', default=6, type=int, 
                    help='topk nums of v3 CL loss')

def main():
    args = parser.parse_args()
    args.conf_ema_range = [float(item) for item in args.conf_ema_range.split(',')]
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    print(args)

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    
    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node: ', ngpus_per_node)
    print('args.gpu: ', args.gpu)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = True
    args.gpu = gpu
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    # if args.gpu is not None:
    #     print("Use GPU: {} for training".format(args.gpu))
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.gpu==0:
        logger = tb_logger.Logger(logdir=os.path.join(args.exp_dir,'tensorboard'), flush_secs=2)
        logging.basicConfig(filename=os.path.join(args.exp_dir,'train_log.log'),
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
        # log the args
        logging.info(args)
    else:
        logger = None

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = PiCO(args, SupConResNet)
    # Unused_idx = [64, 65]
    # print('Unused_idx: ', Unused_idx)
    # for i, (name, param) in enumerate(model.named_parameters()):
    #     if i in Unused_idx:
    #         print(name)
    
    if args.distributed:
        # find_unused_parameters = args.add_bag_head or args.add_seg_head or args.only_moco
        find_unused_parameters = True
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            # print('Use GPU: {} for training'.format(args.gpu))
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            print('total batch_size: ', args.batch_size)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=find_unused_parameters)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=find_unused_parameters)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            
            # if not args.semi_stage2: 
            #     args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            # if not args.semi_stage2:
            #     optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # create dataloader
    args.semi_stage2 = args.semi_start != 10000
    if args.dataset == 'endoscopy':
        if not args.only_bag:
            train_loader, train_givenY, train_sampler, train_dataset, \
                eval_loader_img, eval_dataset_img = load_img_dataset(args, 
                batch_size=args.batch_size, debug=args.debug, transform_path='./configs/transform_new.yaml')
            if args.semi_stage2 or args.only_moco:
                train_loader2, train_givenY2, train_sampler2, train_dataset2, \
                    = load_semi_img_dataset(args, batch_size=args.batch_size, debug=args.debug, transform_path='./configs/transform_new.yaml')
        
                print('traindataset: instances with true labels num: {}/{} ({:.3f}%)'.format(train_dataset2.true_label_num, len(train_dataset2), train_dataset2.true_label_num/len(train_dataset2)*100))
                logging.info('instances with true labels num: {}/{} ({:.3f}%)'.format(train_dataset2.true_label_num, len(train_dataset2), train_dataset2.true_label_num/len(train_dataset2)*100))
        if args.add_bag_head:
            train_loader_bag, train_sampler_bag, train_dataset_bag, \
                eval_loader_bag, eval_sampler_bag, eval_dataset_bag = load_bag_dataset(args, 
                    batch_size=args.batch_size, debug=args.debug, transform_path='./configs/transform_new.yaml')
        if args.add_seg_head:
            train_seg_loader, eval_seg_loader = load_polyp_seg_dataset(args=args)

        print('\n## ---- load all done -----')
        
    else:
        raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")
    # print('## ---- load data done ----')
    # this train loader is the partial label training loader

    if not args.only_bag:
        print('\nCalculating uniform targets...')
        # calculate confidence
        if args.gpu == 0:
            # tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
            # confidence = train_givenY.float()/tempY
            confidence = train_givenY.float()
        else:
            confidence = torch.empty(train_givenY.shape[0], train_givenY.shape[1])
        confidence = confidence.cuda(args.gpu)  # 确保张量在GPU上
        dist.broadcast(confidence, src=0)  # 从rank 0广播

        # set loss functions (with pseudo-targets maintained)
        loss_fn = partial_loss_DDP(confidence, num_class=args.num_class, losstype=args.Cls_LossType)
        if args.distributed and args.gpu is None:
            loss_fn = loss_fn.cuda()
        else:
            loss_fn = loss_fn.cuda(args.gpu)
        loss_cont_fn = SupConLoss_One_CL_Space(temperature=args.temperature, losstype=args.CL_LossType, topk=args.topk)
        # get class distribution aware thresholds
        class_labeled_freq = train_dataset.labeled_freq

    loss_bag = torch.nn.CrossEntropyLoss()
    loss_bag_pseudo1 = torch.nn.CrossEntropyLoss(reduction='none')
    loss_bag_pseudo2 = torch.nn.KLDivLoss(reduction='none')

    # start training
    best_auc = 0
    best_auc_instance = 0
    best_epoch = -1
    best_epoch_instance = -1
    mmc = 0 # mean max confidence
    print('\nStart Training')
    # if args.semi_stage2:
    model.module.reset_queue()
    args.thre_vec_cls = torch.tensor([0.5]*8)
    args.thre_vec_prot = torch.tensor([0.5]*8)
    for epoch in range(args.start_epoch, args.epochs):
        is_best = False
        is_best_instance = False
        start_upd_prot = epoch>=args.prot_start
        start_semi = epoch>=args.semi_start
        start_bag = epoch>=args.sup_bag_start
        start_proto_CL = epoch >= (args.prot_start + 20)

        if epoch == args.prot_start:
            model.module.reset_queue()
        if args.distributed and (not args.only_bag) and args.semi_stage2 and start_semi:
            train_sampler2.set_epoch(epoch)
            # eval_sampler_bag.set_epoch(epoch)
            # train_sampler_bag.set_epoch(epoch)
        adjust_learning_rate(args, optimizer, epoch)
        print('lr: ', round(optimizer.param_groups[0]['lr'],4))


        # generate class-aware thresholds
        if start_semi and epoch % args.dynamic_thre_epoch == 0:
            thre_vec_cls, thre_vec_prot = generate_Class_Aware_thresholds(args, args.batch_size, model, class_labeled_freq, debug=args.debug, transform_path='./configs/transform.yaml')
            logging.info(f'class_labeled_freq: {class_labeled_freq}, thre_vec_cls: {thre_vec_cls}, thre_vec_prot: {thre_vec_prot}')
            model.module.set_thre_vec(thre_vec_cls, thre_vec_prot)
            args.thre_vec_cls = thre_vec_cls
            args.thre_vec_prot = thre_vec_prot
            # if epoch == args.semi_start: # directly set the prototype scores as pseudo_labels
            #     loss_fn.conf_ema_m = 0

        # 1) train & eval on seg task
        if not args.only_bag and args.add_seg_head:
            if args.switch_learning: model.module.switch_learnabel_weights(stage='1')
            train_seg(train_seg_loader, model, optimizer, epoch, args, logger)
            dice = test_seg(eval_seg_loader, model, args)
            print(f'epoch: {epoch}, dice: {dice}')
            logging.info(f'epoch: {epoch}, dice: {dice}')
            if args.gpu==0: logger.log_value('dice', dice, epoch)

        # 2) train & eval on instances task
        if not args.only_bag:
            if args.switch_learning: model.module.switch_learnabel_weights(stage='1')

            if epoch == args.semi_start: 
                print('\nCalculating uniform targets on semi dataset...')
                # calculate confidence
                if args.gpu == 0:
                    # tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
                    # confidence = train_givenY.float()/tempY
                    confidence = train_givenY2.float()
                else:
                    confidence = torch.empty(train_givenY2.shape[0], train_givenY2.shape[1])
                confidence = confidence.cuda(args.gpu)  # 确保张量在GPU上
                dist.broadcast(confidence, src=0)  # 从rank 0广播

                # set loss functions (with pseudo-targets maintained)
                loss_fn = partial_loss_DDP(confidence, num_class=args.num_class, losstype=args.Cls_LossType)
                if args.distributed and args.gpu is None:
                    loss_fn = loss_fn.cuda()
                else:
                    loss_fn = loss_fn.cuda(args.gpu)
            if args.only_moco:
                train_only_moco(train_loader2, model, loss_fn, loss_cont_fn, loss_bag_pseudo1, optimizer, epoch, args, logger, start_upd_prot, start_semi, start_bag, start_proto_CL)
            else:
                train(train_loader2 if start_semi else train_loader, model, loss_fn, loss_cont_fn, loss_bag_pseudo1, loss_bag_pseudo2, optimizer, epoch, args, logger, start_upd_prot, start_semi, start_bag, start_proto_CL)
            if start_semi: loss_fn.set_conf_ema_m(epoch, args)

            # eval on instances
            # if (epoch + 1) % 100 == 0:
            result_img = test_img(eval_loader_img, model, args)
            print('epoch:', epoch, ',' ,result_img)
            logging.info(f'epoch: {epoch}, result_img: {result_img}')
            
            auc_instance = result_img['micro_cls']['auc']
            
            if args.gpu==0: logger.log_value('auc_instance', auc_instance, epoch)

            if auc_instance > best_auc_instance:
                best_epoch_instance = epoch
                best_auc_instance = auc_instance
                auc_instance_prot = result_img['micro_prot']['auc']
                is_best_instance = True

            print(f'best_epoch: {best_epoch_instance}, best_auc_cls: {best_auc_instance:.3f}, auc_prot: {auc_instance_prot:.3f}')
            # save ckp
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best_instance, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
            best_file_name='{}/checkpoint_bestInstance_{}_{}.pth.tar'.format(args.exp_dir,epoch,auc_instance))

        # 3) train & eval on bags task
        if args.add_bag_head and start_bag: # and ((epoch + 1) % 100 == 0):
            if args.switch_learning: model.module.switch_learnabel_weights(stage='2')
            # for i in range(0, 10):
            #     epoch_bag = round(epoch+i*0.1, 2)
            #     # if args.sample_balance:
            #     train_dataset_bag.random_choose_nocancer_img()
            #     train_bag_accumulate(train_loader_bag, model, loss_bag, optimizer, epoch_bag, args)
            if args.sample_balance: train_dataset_bag.balance_label_sample()
            train_bag_accumulate(train_loader_bag, model, loss_bag, optimizer, epoch, args, tb_logger=logger)

            metrics = test_bag(eval_loader_bag, model, args)
            print('epoch:', epoch, ',' , metrics)
            logging.info(f'epoch: {epoch}, result_bag: {metrics}')

            auc = metrics['AUC']
            if args.gpu==0: logger.log_value('auc_bag', auc, epoch)
            
            if auc > best_auc:
                best_epoch = epoch
                best_auc = auc
                is_best = True
            print(f'epoch: {best_epoch}, best_auc: {best_auc}')

            # save ckp
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
            best_file_name='{}/checkpoint_bestBag_{}_{}.pth.tar'.format(args.exp_dir,epoch,auc))
        

## ------------------------------------------##
def train(train_loader, model, loss_fn, loss_cont_fn, loss_bag_pseudo1, loss_bag_pseudo2, optimizer, epoch, args, tb_logger, start_upd_prot=False, start_semi=False, start_bag=False, start_proto_CL=False):
    print('\n ==> train on instances...')     
    batch_time = AverageMeter('Batch_Time', ':1.2f')
    data_time = AverageMeter('Data_time', ':1.2f')
    # acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    # acc_proto = AverageMeter('Acc@Proto', ':2.2f')
    loss_bag_log = AverageMeter('Loss@1-bag', ':2.2f')
    loss_cls_log = AverageMeter('Loss@Cls', ':2.2f')
    loss_cont_sup_log = AverageMeter('Loss@Cont_sup', ':2.2f')
    loss_cont_moco_log = AverageMeter('Loss@Cont_moco', ':2.2f')
    loss_cont_proto_log = AverageMeter('Loss@Cont_proto', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_cls_log, loss_cont_sup_log, loss_cont_moco_log, loss_cont_proto_log, loss_bag_log],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    
    end = time.time()

    for i, pack in enumerate(train_loader):
        loss_dict = {}
        images_w, images_s, labels, true_labels, bag_labels, index = pack['image_w'], pack['image_s'], pack['label'], pack['label_true'], pack['label_corresponding_patient'], pack['index']
        label_true_8class = pack['label_true_8class']
        # labels: [1,0,0] or [0,1,0] or [1,1,1](unkonwn)
        # true_labels: 0 or 1 or -1(unknown)
        # measure data loading time
        data_time.update(time.time() - end)
        X_w, X_s, Y, index = images_w.cuda(), images_s.cuda(), label_true_8class.cuda(), index.cuda()
        # print('Y: ', Y)
        # print('confidence: ', loss_fn.confidence[index])
        # Y_true = true_labels.long().detach().cuda()
        # for showing training accuracy and will not be used when training
        cls_out, features_cont, pseudo_target_cont, score_prot_b, logits_pseudo_bag, mask_valid_prot, logits_pseudo_bag_k = model(X_w, X_s, Y, args, start_upd_prot=start_upd_prot, semi_stage2=start_semi, task='img')
        if epoch % 3 ==0 and i == 0 and epoch > 200:
            os.makedirs(f'./{args.exp_dir}/save_npy', exist_ok=True)
            np.save(f'./{args.exp_dir}/save_npy/{epoch}_features_cont.npy', 
                    features_cont.detach().clone().cpu().numpy())
            np.save(f'./{args.exp_dir}/save_npy/{epoch}_pseudo_target_cont.npy', 
                        pseudo_target_cont.detach().clone().cpu().numpy())
        # print('Y: ', Y)
        # print('pseudo_target_cont: ', pseudo_target_cont)
        # print('feat', features_cont.max(), features_cont.min())
        # cls_out: [b,C], features_cont: [b,D], pseudo_target_cont: [L,C], score_prot: [b,C]
        batch_size = cls_out.shape[0]
        num_class = cls_out.shape[1]
 
        '''the pos/neg samples has been splited with CLS-1 and maybe [true label], for contrastive loss'''
        if start_upd_prot and start_semi:
            '''use maybe [true labels] to update confidence'''
            loss_fn.confidence_update(temp_un_conf=score_prot_b, batch_index=index, batchY=Y, true_labels=true_labels)
            # warm up ended

        # if start_upd_prot:
        #     '''use maybe [true labels] to update confidence'''
        #     loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=Y, true_labels=true_labels)
        #     # warm up ended
       
        if args.only_moco or not start_upd_prot:
            mask = None
            loss_dict['loss_cont_moco'] = loss_cont_fn(features=features_cont, mask=mask, batch_size=batch_size)
        elif start_upd_prot:
            # mask = torch.zeros((num_class, batch_size, pseudo_target_cont.shape[0])) #[B*C,B*C]
            # for cls_i in range(num_class):
            #     pseudo_target_cont_class = pseudo_target_cont[:,cls_i].contiguous().view(-1,1)
            #     mask_ = torch.eq(pseudo_target_cont_class[:batch_size], pseudo_target_cont_class.T).float()
            #     mask[cls_i] = mask_
            # mask = mask.cuda() #[C,B,B]
            mask = compute_jaccard_similarity_matrix(pseudo_target_cont[:batch_size], pseudo_target_cont, label_enhancing=args.label_enhancing).cuda()
            if args.no_queue:
                features_cont = features_cont[:batch_size]
                mask = mask[:,:batch_size]
            loss_dict['loss_cont_sup'] = loss_cont_fn(features=features_cont, mask=mask, batch_size=batch_size)

        if args.proto_cont and start_proto_CL:
            prototype_labels = torch.eye(num_class+1).to(pseudo_target_cont.device)[:,:8]
            cont_feature = torch.cat([features_cont[:batch_size], model.module.prototypes.detach()], dim=0) #[B+C+1,D]
            cont_label = torch.cat([pseudo_target_cont[:batch_size], prototype_labels], dim=0) #[B+C+1,C]
            mask = compute_jaccard_similarity_matrix(pseudo_target_cont[:batch_size], cont_label, label_enhancing=False).cuda()
            loss_dict['loss_cont_proto'] = loss_cont_fn(features=cont_feature, mask=mask, batch_size=batch_size)

        # Valid when SupCL work and keep_moco=True
        if args.keep_moco and mask is not None:
            loss_dict['loss_cont_moco'] = loss_cont_fn(features=features_cont, mask=None, batch_size=batch_size)
        
        
        # classification loss
        loss_cls = loss_fn(cls_out, index, mask_valid_prot)
        loss_cls_log.update(loss_cls.item())

        # pseudo bag loss
        if args.pseudo_bag_sup and start_bag:
            loss_pseudo_bag = One_Bag_Loss(logits_pseudo_bag, logits_pseudo_bag_k, loss_fn.confidence[index, :], bag_labels.to(logits_pseudo_bag.device), loss_bag_pseudo1, loss_bag_pseudo2)
            loss_bag_log.update(loss_pseudo_bag.item())
            loss_dict['loss_pseudo_bag'] = loss_pseudo_bag

        # loss = loss_cls + args.loss_weight * (sum[loss_dict[key] for key in loss_dict.keys()])+ loss_dummy
        loss = loss_cls + args.loss_weight * (sum([loss_dict[key] for key in loss_dict.keys()]))
        # if args.gpu == 0 and i % (len(train_loader)//10) == 0 and i != 0:
        #     print('Epoch: {}/{}, Batch: {}/{}, loss: {:.4f}'.format(epoch, args.epochs, i, len(train_loader), loss.item()))
        
        if 'loss_cont_sup' in loss_dict.keys():
            loss_cont_sup_log.update(loss_dict['loss_cont_sup'].item()) 
        if 'loss_cont_moco' in loss_dict.keys():
            loss_cont_moco_log.update(loss_dict['loss_cont_moco'].item()) 
        if 'loss_cont_proto' in loss_dict.keys():
            loss_cont_proto_log.update(loss_dict['loss_cont_proto'].item())


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # check_gradient_layout(model)
        optimizer.step()
        # measure elapsed time

        batch_time.update(time.time() - end)
        end = time.time()

        if i % (len(train_loader)//10 + 1) == 0:      
            
            progress.display(i)
            # logging.info('Epoch: {}/{}, Batch: {}/{}, loss_cls: {:.4f}, loss_cont: {:.4f}'.format(epoch, args.epochs, i, len(train_loader), loss_cls.item(), loss_cont.item()))
            logging.info('Epoch: {}/{}, Batch: {}/{}, loss_cls: {:.4f}, loss_one_bag: {:.4f}, loss_cont_sup: {:.4f}, loss_cont_moco: {:.4f}, loss_cont_proto: {:.4f}'\
                .format(epoch, args.epochs, i, len(train_loader), loss_cls.item(), \
                    loss_dict['loss_pseudo_bag'].item() if 'loss_pseudo_bag' in loss_dict.keys() else 0, \
                    loss_dict['loss_cont_sup'].item() if 'loss_cont_sup' in loss_dict.keys() else 0, \
                    loss_dict['loss_cont_moco'].item() if 'loss_cont_moco' in loss_dict.keys() else 0, \
                    loss_dict['loss_cont_proto'].item() if 'loss_cont_proto' in loss_dict.keys() else 0
                                ))
    
    if args.stable_queue and start_upd_prot: 
        print('queue:', model.module.Queue_analyze_distribution().round(2)) 
        logging.info(f'queue: {model.module.Queue_analyze_distribution().round(2)}') 
        
    if args.gpu == 0:
        # tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        # tb_logger.log_value('Prototype Acc', acc_proto.avg, epoch)
        tb_logger.log_value('Classification Loss', loss_cls_log.avg, epoch)
        tb_logger.log_value('1-bag Loss', loss_bag_log.avg, epoch)
        tb_logger.log_value('Contrastive Loss_sup', loss_cont_sup_log.avg, epoch)
        tb_logger.log_value('Contrastive Loss_moco', loss_cont_moco_log.avg, epoch)
        tb_logger.log_value('Contrastive Loss_proto', loss_cont_proto_log.avg, epoch)

    return None

def train_only_moco(train_loader, model, loss_fn, loss_cont_fn, loss_bag_c, optimizer, epoch, args, tb_logger, start_upd_prot=False, start_semi=False, start_bag=False, start_proto_CL=False):
    print('\n ==> train on instances - onlyMoCo...')     
    batch_time = AverageMeter('Batch_Time', ':1.2f')
    data_time = AverageMeter('Data_time', ':1.2f')
    # acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    # acc_proto = AverageMeter('Acc@Proto', ':2.2f')
    loss_cont_moco_log = AverageMeter('Loss@Cont_moco', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_cont_moco_log],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    
    end = time.time()

    for i, pack in enumerate(train_loader):
        loss_dict = {}
        images_w, images_s, labels, true_labels, bag_labels, index = pack['image_w'], pack['image_s'], pack['label'], pack['label_true'], pack['label_corresponding_patient'], pack['index']
        label_true_8class = pack['label_true_8class']
        # labels: [1,0,0] or [0,1,0] or [1,1,1](unkonwn)
        # true_labels: 0 or 1 or -1(unknown)
        # measure data loading time
        data_time.update(time.time() - end)
        X_w, X_s, Y, index = images_w.cuda(), images_s.cuda(), label_true_8class.cuda(), index.cuda()

        cls_out, features_cont, pseudo_target_cont, score_prot_b, logits_pseudo_bag, mask_valid_prot = model(X_w, X_s, Y, args, start_upd_prot=start_upd_prot, semi_stage2=start_semi, task='img')

        batch_size = cls_out.shape[0]
        num_class = cls_out.shape[1]
 
        loss_cont_moco = loss_cont_fn(features=features_cont, mask=None, batch_size=batch_size)
        loss_cont_moco_log.update(loss_cont_moco.item())
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_cont_moco.backward()
        # check_gradient_layout(model)
        optimizer.step()
        # measure elapsed time

        batch_time.update(time.time() - end)
        end = time.time()

        if i % (len(train_loader)//10 + 1) == 0:      
            
            progress.display(i)
            # logging.info('Epoch: {}/{}, Batch: {}/{}, loss_cls: {:.4f}, loss_cont: {:.4f}'.format(epoch, args.epochs, i, len(train_loader), loss_cls.item(), loss_cont.item()))
            logging.info('Epoch: {}/{}, Batch: {}/{}, loss_cont_moco: {:.4f}'\
                .format(epoch, args.epochs, i, len(train_loader), loss_cont_moco.item()))
    
    if args.gpu == 0:
        tb_logger.log_value('Contrastive Loss_moco', loss_cont_moco_log.avg, epoch)

    return None


def train_bag_accumulate(train_loader, model, loss_bag, optimizer, epoch, args, accumulation_steps=20, tb_logger=None):
    print('\n ==> train on bag...')     
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    batch_time = AverageMeter('Batch_Time', ':1.2f')
    data_time = AverageMeter('Data_time', ':1.2f')
    loss_bag_log = AverageMeter('Loss@Cls_bag', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_bag_log],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    for i, pack in enumerate(train_loader):  
        data_time.update(time.time() - end)
        X, Y = pack['bag'], pack['label_bag']  # X: (1,3,352,352)*N; Y: (1,)
        # print(len(X), X[0].shape, len(Y), Y.shape)
        X = torch.cat(X, dim=0) # [N,3,352,352]
        # print(X.shape, Y.shape)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            bag_logit, importance_scores_raw, logits_pre = model(X.cuda(), args=args, task='bag') # 2
            # print(prof.key_averages().table(sort_by="cuda_time_total"))
            bag_logit = torch.softmax(bag_logit,dim=1)
            # print(bag_pred.shape, Y.shape) # [b,C], [C]
            loss_res = loss_bag(bag_logit,Y.cuda())
            loss_atten = attention_constrain_loss(logits_pre, Y, importance_scores_raw)
        # t_f = time.time()
        # print('0-forward time:', t_f-end)
        weight_bag_loss = args.loss_weight_bag
        bag_loss = (loss_res+loss_atten) * weight_bag_loss
        loss_bag_log.update(bag_loss.item())
        # print loss
        if i % (len(train_loader)//10+1) == 0:
            # print('Epoch: {}/{}, Batch: {}/{}, loss: {:.4f}'.format(epoch, args.epochs, i, len(train_loader), bag_loss.item()))
            logging.info('Epoch: {}/{}, Batch: {}/{}, loss: {:.4f}'.format(epoch, args.epochs, i, len(train_loader), bag_loss.item()))
            progress.display(i)
        # compute gradient and do SGD step
        bag_loss = bag_loss / accumulation_steps
        scaler.scale(bag_loss).backward()  # 缩放梯度
        if (i+1) % accumulation_steps == 0 or i == (len(train_loader) - 1):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        # optimizer.zero_grad()
        # bag_loss.backward()
        # optimizer.step()
        # print('0-backward time:', time.time()-t_f)
        batch_time.update(time.time() - end)
        end = time.time()
        # print('0-batch time:', batch_time)
    if args.gpu == 0:
        # tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        # tb_logger.log_value('Prototype Acc', acc_proto.avg, epoch)
        tb_logger.log_value('Bag Classification Loss', loss_bag_log.avg, epoch)

def test_bag(eval_loader_bag, model, args):
    print('\n ==> Evaluation on bag...')     
    model.eval()
    bag_preds, bag_logits, bag_labels = [], [], []
    with torch.no_grad():
        for i, pack in enumerate(eval_loader_bag):  
            # if i % (len(eval_loader_bag)//5) == 0 and i != 0: print(i)
            X, Y = pack['bag'], pack['label_bag'] # X: (1,3,352,352)*N; Y: (1,)
            X = torch.cat(X, dim=0) # [N,3,352,352]
            bag_logit,A_raw,score_clsHead = model(X, args=args, task='bag') # [1,C]
            # print(bag_logit)
            bag_logit = torch.softmax(bag_logit,dim=1)
            bag_pred = torch.argmax(bag_logit, dim=1) # 1

            bag_preds.append(bag_pred.detach())
            bag_logits.append(bag_logit[:,1].detach())
            bag_labels.append(Y.cuda())
    
    bag_preds_concat = concat_all_gather(torch.cat(bag_preds, dim=0)).cpu().numpy()
    bag_logits_concat = concat_all_gather(torch.cat(bag_logits, dim=0)).cpu().numpy()
    bag_labels_concat = concat_all_gather(torch.cat(bag_labels, dim=0)).cpu().numpy()
    
    metrics = calculate_metrics(bag_labels_concat, bag_preds_concat, bag_logits_concat)
    return metrics


def test_img(eval_loader_img, model, args):
    print('\n ==> Evaluation on img...')  
    outputs_cls, outputs_prot = [], []
    labels = []
    model.eval()
    with torch.no_grad():
        for i, pack in enumerate(eval_loader_img):
            X = pack['image_eval']
            labels.append(pack['label_true_8class'].view(1, -1).cuda())
            # orders.append(pack['idx_unlabeled'].cpu())
            score_clsHead, _, _, score_prot = model(X, args=args, eval_only=True, task='img')
            outputs_cls.append(score_clsHead.view(1, -1).detach()) # [B,C]
            outputs_prot.append(score_prot.view(1, -1).detach())

    outputs_cls = torch.cat(outputs_cls, dim=0)
    outputs_prot = torch.cat(outputs_prot, dim=0)
    labels = torch.cat(labels, dim=0)
    outputs_cls_concat = concat_all_gather(outputs_cls)
    outputs_prot_concat = concat_all_gather(outputs_prot)
    labels_concat = concat_all_gather(labels)
    # result = evaluate_multilabel_micro_auc(labels, outputs_cls, outputs_prot)
    # result = evaluate_multilabel_micro_auc(labels_concat.cpu().numpy(), outputs_cls_concat.cpu().numpy(), outputs_prot_concat.cpu().numpy())
    # print(outputs_cls_concat.shape, outputs_prot_concat.shape, labels_concat.shape)
    result = calculate_multilabel(labels_concat.cpu().numpy(), outputs_cls_concat.cpu().numpy(), outputs_prot_concat.cpu().numpy())
    
    return result

def train_seg(train_loader, model, optimizer, epoch, args, tb_logger):
    print('\n ==> train on seg dataset...')    
    model.train()
    batch_time = AverageMeter('Batch_Time', ':1.2f')
    data_time = AverageMeter('Data_time', ':1.2f')
    loss_seg_log = AverageMeter('Loss@seg', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_seg_log],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    for i, (images, gts) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images, gts = images.cuda(), gts.float().unsqueeze(1).cuda()
        # ---- forward ----
        P1, P2= model(images, args=args, task='seg')
        # ---- loss function ----
        loss_P1 = structure_loss(P1, gts)
        loss_P2 = structure_loss(P2, gts)
        loss = loss_P1 + loss_P2 
        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_seg_log.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        # print loss    
        if i % (len(train_loader)//10) == 0 and i != 0:
            logging.info('Epoch: {}/{}, Batch: {}/{}, loss_seg: {:.4f}'.format(epoch, args.epochs, i, len(train_loader), loss.item()))
            progress.display(i)

    if args.gpu == 0:
        # tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        # tb_logger.log_value('Prototype Acc', acc_proto.avg, epoch)
        tb_logger.log_value('Seg Loss', loss_seg_log.avg, epoch)

def test_seg(eval_loader_seg, model, args):
    print('\n ==> Evaluation on seg dataset...')     
    model.eval()
    DSC = 0.0
    num1 = len(eval_loader_seg)
    with torch.no_grad():
        for i, (images, gts) in enumerate(eval_loader_seg):
            images, gts = images.cuda(), gts.float().unsqueeze(1)
            # ---- forward ----
            
            P1, P2 = model(images, args=args, task='seg')
            # print('P1', P1.shape, 'gts', gts.shape)
            res = F.upsample(P1 + P2 , size=gts.shape[-2:], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            input = res
            target = np.array(gts)
            # N = gts.shape
            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            DSC = DSC + dice

    return DSC / num1
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if state['epoch'] % 100 == 0:
        shutil.copyfile(filename, filename.replace('checkpoint', 'checkpoint_'+str(state['epoch'])))
        print(f'save checkpoint')
    if is_best:
        shutil.copyfile(filename, best_file_name)

if __name__ == '__main__':
    main()
